import os, torch, json, yaml, argparse
num_gpus = torch.cuda.device_count()
os.environ["CUDA_VISIBLE_DEVICES"] = f"{num_gpus - 1}"
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from datetime import datetime
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainerCallback
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from train.utils.decision.decision_dataset import DecisionDatasetGRPO
import numpy as np
import re
import wandb
from dotenv import load_dotenv, find_dotenv
from functools import partial

import train.utils.decision.eval_states as eval_states
from llm_mpc import RaceLLMMPC as RaceLLM 
from tests.decision_tester.decision_tester import DecisionTester
from inference.inf_pipeline import RaceLLMPipeline

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") 
OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN") 
WANDB_API_KEY = os.getenv("WANDB_API_KEY", None) # Optional, can be set in the .env file
race_llm = RaceLLM(openai_token=OPENAI_API_TOKEN, model='gpt-4o', no_ROS=True)

class PeriodicEvalCallback(TrainerCallback):
    def __init__(self, eval_fn, model, tokenizer, interval=50):
        self.eval_fn = eval_fn
        self.model = model
        self.tokenizer = tokenizer
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval == 0 and state.global_step != 0:
            print(f"[EvalCallback] Running evaluation at step {state.global_step}...")
            self.eval_fn(self.model, self.tokenizer, step=state.global_step)

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Initialize wandb with proper login
def init_wandb():
    wandb_api_key = WANDB_API_KEY
    
    if not wandb_api_key:
        print("WANDB_API_KEY not found in environment variables.")
        print("Please provide your wandb API key or press Enter to skip wandb logging:")
        user_input = input()
        
        if user_input.strip():
            wandb_api_key = user_input.strip()
        else:
            print("Skipping wandb logging.")
            return False
    
    try:
        wandb.login(key=wandb_api_key)
        print("Successfully logged in to Weights & Biases.")
        return True
    except Exception as e:
        print(f"Failed to log in to Weights & Biases: {e}")
        print("Continuing without wandb logging.")
        return False

def chat_mapping(chat_template="phi-3"):
    if chat_template == "phi-3":
        return {"role": "from", "content": "value", "user": "human", "assistant": "gpt"}
    elif chat_template == "qwen-2.5":
        return {"role": "role", "content": "content", "user": "user", "assistant": "assistant"}
    else:
        raise ValueError(f"Chat template {chat_template} not recognized. Please use 'phi-3' or 'qwen-2.5'.")

def train(cfg):
    out_dir = cfg["training"]["out_dir"]
    chat_template = cfg["training"]["chat_template"]
    use_rag = cfg["training"]["use_rag"]
    base_model = cfg["model"]["base_model"]
    load_in_4bit = cfg["model"]["load_in_4bit"]
    train_bool = cfg["training"]["train_bool"]
    lora_rank = cfg["training"]["lora_rank"]
    wandb_project = cfg["wandb"]["project"]
    max_seq_length = cfg["model"]["max_seq_length"]
    experiment_name = base_model.split("/")[-1] + "_GRPO_"
    experiment_name += datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    use_wandb = init_wandb()
    report_to = "wandb" if use_wandb else "none"
    
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(project=wandb_project, config=cfg)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=cfg["model"]["fast_inference"],
        max_lora_rank=lora_rank,
        gpu_memory_utilization=cfg["model"]["gpu_memory_utilization"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=cfg["model"]["target_modules"],
        lora_alpha=cfg["model"]["lora_alpha"],
        use_gradient_checkpointing=cfg["model"]["use_gradient_checkpointing"],
        random_state=cfg["model"]["random_state"],
    )
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template,
        mapping=chat_mapping(),
    )

    ####################################REWARD FUNCTION####################################

    step_state = {"step": 0}

    def wrapped_correctness_reward(*args, **kwargs):
        kwargs["step"] = step_state["step"]
        return correctness_reward_func(*args, **kwargs)

    def wrapped_format_reward(*args, **kwargs):
        kwargs["step"] = step_state["step"]
        return format_reward_func(*args, **kwargs)
    
    def compute_format_score(completion, tag):
        open_tags = completion.count(f"<{tag}>")
        close_tags = completion.count(f"</{tag}>")

        if open_tags == close_tags == 1:
            return 1.0
        else:
            return 0.0

    def format_reward_func(prompts, completions, **kwargs) -> list[float]:
        step = kwargs.get("step", 0)
        max_steps = cfg["grpo"]["max_steps"]

        print(f"format_reward_func: step {step} of {max_steps}=================================================================")
        reasoning_reward = np.zeros(len(prompts))
        explanation_reward = np.zeros(len(prompts))
        answer_reward = np.zeros(len(prompts))
        for i in range(len(prompts)):
            reasoning_reward[i] = compute_format_score(completions[i], "reasoning")
            explanation_reward[i] = compute_format_score(completions[i], "explanation")
            answer_reward[i] = compute_format_score(completions[i], "answer")

        reward = (reasoning_reward + explanation_reward + answer_reward) / 3.0
        print(f"format_reward_func: {reward}")
        return [reward]

    def correctness_reward_func(prompts, completions, robot_state, evaluation_function, **kwargs) -> list[float]:
        #print("PROMPT 0", prompts[0])
        print("COMPLETION 0", completions[0])
        print("ROBOT STATES 0", robot_state[0])
        print("EVALUATION FUNCTION 0", evaluation_function[0])
        eval_func = getattr(eval_states, evaluation_function[0])
        # Compute the reward based on the evaluation function
        gt = eval_func(robot_state)
        print("gt:", gt)
        
        # Check what the model predicted by parsing what it says after Adhering to Human:
        pred = []
        for i in range(len(prompts)):
            match = re.search(r'Adhering to Human:\s*(True|False)', completions[i])
            if match:
                pred_val = match.group(1)
                pred.append(True if pred_val == "True" else False)
            else:
                pred.append(None)

        print("pred:",pred)

        # Compute the number of correct predictions doing something like correct = 1 if gt == pred else 0
        correct = []
        gt_list = eval_func(robot_state)
        for gt_val, pred_val in zip(gt_list, pred):
            if pred_val is None:
                correct.append(0)  # format violation → wrong
            else:
                correct.append(1 if gt_val == pred_val else 0)

        print("correct:", correct)
        
        # Log output token length to wandb
        if use_wandb:
            token_lengths = [len(tokenizer.tokenize(completion)) for completion in completions]
            avg_token_len = np.mean(token_lengths)
            wandb.log({
                "output_token_lengths": {
                    "avg": avg_token_len,
                    "min": np.min(token_lengths),
                    "max": np.max(token_lengths),
                }
            })

        return [correct]

    # Custom wandb callback
    class WandbCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and use_wandb:
                wandb.log(logs)

    class StepTrackerCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            step_state["step"] = state.global_step



    dataset = DecisionDatasetGRPO(
        raw_robot_states_dir=cfg["dataset"]["raw_robot_states_dir"],
        test_cases=eval_states.EVAL_CASES,
        use_rag=use_rag,
        from_raw=cfg["dataset"]["from_raw"],
        index=race_llm.decision_index,
        mem_nb=cfg["dataset"]["mem_nb"],
        shuffle=cfg["dataset"]["shuffle"],
    )

    print(f"Training with {len(dataset)} samples.")

    if train_bool:
        training_args = GRPOConfig(
            use_vllm=True,
            learning_rate=cfg["grpo"]["learning_rate"],
            adam_beta1=cfg["grpo"]["adam_beta1"],
            adam_beta2=cfg["grpo"]["adam_beta2"],
            weight_decay=cfg["grpo"]["weight_decay"],
            warmup_ratio=cfg["grpo"]["warmup_ratio"],
            lr_scheduler_type=cfg["grpo"]["lr_scheduler_type"],
            optim=cfg["grpo"]["optim"],
            logging_steps=cfg["grpo"]["logging_steps"],
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            per_device_train_batch_size=cfg["grpo"]["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["grpo"]["gradient_accumulation_steps"],
            num_generations=cfg["grpo"]["num_generations"],
            max_prompt_length=cfg["grpo"]["max_prompt_length"],
            max_completion_length=cfg["grpo"]["max_completion_length"],
            max_steps=cfg["grpo"]["max_steps"],
            save_steps=cfg["grpo"]["save_steps"],
            max_grad_norm=0.1,
            report_to=report_to,
            output_dir=f"{out_dir}/{experiment_name}",
        )
        
        # Add custom callbacks if enabled
        eval_fn = partial(evaluate_model, use_rag=use_rag, chat_template=chat_template, max_step=cfg["grpo"]["max_steps"])
        assert cfg["grpo"]["max_steps"] % cfg["evaluation"]["eval_interval_steps"] == 0, "max_steps must be divisible by eval_interval_steps"

        callbacks = [
            WandbCallback(),
            PeriodicEvalCallback(eval_fn=eval_fn, model=model, tokenizer=tokenizer, interval=cfg["evaluation"]["eval_interval_steps"]),
            StepTrackerCallback(),
        ]

        trainer = GRPOTrainer(
            model = model,
            processing_class = tokenizer,
            reward_funcs = [
                wrapped_correctness_reward,
                wrapped_format_reward
            ],
            args = training_args,
            train_dataset = dataset,
            callbacks = callbacks,  # Add callbacks here
        )

        # GPU stats
        # Get properties of the GPU device at index 0
        gpu_stats = torch.cuda.get_device_properties(0)
        # Get the maximum reserved GPU memory in GB and round to 3 decimal places
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        # Get the total GPU memory in GB and round to 3 decimal places
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        # Display the GPU name and maximum memory
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.") 
        # Display the reserved memory amount
        print(f"{start_gpu_memory} GB of memory reserved.")
        
        # Log GPU info to wandb if enabled
        if use_wandb:
            wandb.log({
                "gpu_info": {
                    "name": gpu_stats.name,
                    "max_memory_gb": max_memory,
                    "reserved_memory_gb": start_gpu_memory
                }
            })

        trainer_stats = trainer.train()
        
        # Log final training stats to wandb if enabled
        if use_wandb:
            wandb.log({
                "training_complete": True,
                "training_stats": trainer_stats
            })

    model.save_pretrained(f"{out_dir}/{experiment_name}")
    print("Training complete, saved model: ", f"{out_dir}/{experiment_name}")
    
    if use_wandb:
        # Finish wandb run
        wandb.finish()
    
    return model, tokenizer

def evaluate_model(model, tokenizer,
                   use_rag=True, chat_template="qwen-2.5", max_step=1e3, step=0):
    
    print("Evaluating on full dataset.")
    mini = False
    
    evaluator = DecisionTester(
        model_name="GRPO_Training",
        all_tests=True,
        mini=mini, # full if it is the final step
        local=True,
        use_rag=use_rag,
        quant=False,
        OPENAI_API_TOKEN=OPENAI_API_TOKEN,
    )
    
    llm = RaceLLMPipeline(model_dir=None, model=model, tokenizer=tokenizer, chat_template=chat_template)
    
    base_data_dir = "/embodiedai/tests/decision_tester/robot_states"
    possible_datasets = [filename for filename in os.listdir(base_data_dir) if filename.endswith(".json")]
    
    all_results = {}
    for i, dataset in enumerate(possible_datasets):
        data_dir = os.path.join(base_data_dir, dataset)
        # Evaluate the decision making
        results = None
        try:            
            results = evaluator.eval_decision_making(
                llm=llm,
                data_dir=data_dir,
                data_name=dataset
            )
        except Exception as e:
            print(f"Error evaluating dataset {dataset}: {e}")
            continue

        if wandb.run is not None and results is not None:
             # Extract metrics
            acc = results["overall_accuracy"]
            case_accuracies = results["case_accuracies"]
            log_file = results["log_file"]
            
            # Log scalar metrics
            wandb.log({
                f"eval/{dataset}/overall_accuracy": acc,
            })

            # Log per-case accuracy
            for case, val in case_accuracies.items():
                wandb.log({f"eval/{dataset}/case_accuracy/{case}": val})

            # Log file as artifact (correct way)
            artifact = wandb.Artifact(name=f"eval_log_{dataset}_step{step}", type="evaluation")
            artifact.add_file(log_file)
            wandb.run.log_artifact(artifact)

            all_results[dataset] = results
        
    print("#################Final Eval###################")
    print(all_results)
    # log to wandb
    if wandb.run is not None and all_results:
        wandb.log({
            f"eval/overall_accuracy": sum([results['overall_accuracy'] for results in all_results.values()]) / len(all_results),
        })
    print("#################End of Eval###################")

    return all_results
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, 
                        default="train/config/rl_decision_train.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    train(load_config(args.config))
