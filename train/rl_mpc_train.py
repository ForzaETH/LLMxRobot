import os, torch, yaml, argparse, time
num_gpus = torch.cuda.device_count()
os.environ["CUDA_VISIBLE_DEVICES"] = f"{num_gpus - 1}"
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from datetime import datetime
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainerCallback
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
import numpy as np
import wandb
from dotenv import load_dotenv, find_dotenv
from functools import partial
import roslibpy

from train.utils.mpc.mpc_dataset import MPCDatasetGRPO
import train.utils.mpc.eval_driving as eval_driving
from llm_mpc import RaceLLMMPC as RaceLLM 
from tests.mpc_tester.mpc_tester import TrainingTester as MPCTester

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", None) # Optional, can be set by user in .env file
MPC_PARAM_NAMES = ["qv", "qn", "qalpha", "qac", "qddelta", "alat_max",
                   "a_min", "a_max", "v_min", "v_max", "track_safety_margin"]

class PeriodicEvalCallback(TrainerCallback):
    def __init__(self, eval_fn, shared_ros, model, tokenizer, host_ip, interval=50, run_name="no_wandb"):
        self.eval_fn = eval_fn
        self.shared_ros = shared_ros
        self.model = model
        self.tokenizer = tokenizer
        self.interval = interval
        self.host_ip = host_ip
        self.run_name = run_name

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval == 0 and state.global_step != 0:
            print(f"[EvalCallback] Running evaluation at step {state.global_step}...")
            self.eval_fn(self.model, self.shared_ros, self.tokenizer, self.host_ip, step=state.global_step)

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
    # Get wandb run name
    run_name = wandb.run.name if use_wandb else "no_wandb"
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
    
    # Init the race_llm
    shared_ros = roslibpy.Ros(host=cfg["training"]["ros_host_ip"], port=9090)
    shared_ros.run()
    race_llm = RaceLLM(openai_token=OPENAI_API_TOKEN, 
                       model='training', 
                       ros=shared_ros, 
                       host_ip=cfg["training"]["ros_host_ip"], 
                       no_ROS=False)


    ####################################REWARD FUNCTION####################################

    step_state = {"step": 0}

    def wrapped_driving_reward(*args, **kwargs):
        kwargs["step"] = step_state["step"]
        return driving_adaption_reward(*args, **kwargs)

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
        reasoning_reward = np.zeros(len(prompts))
        answer_reward = np.zeros(len(prompts))
        for i in range(len(prompts)):
            reasoning_reward[i] = compute_format_score(completions[i], "reasoning")
            answer_reward[i] = compute_format_score(completions[i], "answer")

        reward = (reasoning_reward + answer_reward) / 2.0
        print(f"format_reward_func: {reward}")
        return [reward]

    def mpc_param_extraction_reward(prompts, completions, evaluation_function, baseline_rmse, **kwargs) -> list[float]:
        rewards = np.zeros(len(prompts))
        for i in range(len(prompts)):
            # Extract the MPC Params 
            extracted_command, _ = race_llm._sanitize_tune_output(completions[i])
            if extracted_command is None:
                rewards[i] = 0.0
                continue
            else:
                rewards[i] = 1 # Add a base reward if the command is valid
        print(f"mpc_param_extraction_reward = {rewards}")
        return [rewards]
    
    def mpc_param_name_reward(prompts, completions, evaluation_function, baseline_rmse, **kwargs) -> list[float]:
        rewards = np.zeros(len(prompts))
        for i in range(len(prompts)):
            extracted_command, _ = race_llm._sanitize_tune_output(completions[i])
            # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            # print(f"EXTRACTED COMMAND: {extracted_command}")
            # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            if extracted_command is None:
                rewards[i] = 0.0
                continue
            else:
                correct_params = 0
                partially_correct_params = 0
                if len(extracted_command) == 0:
                    rewards[i] = 1
                else:
                    for predicted_param_name in extracted_command.keys():
                        # Check if param name is a string
                        if not isinstance(predicted_param_name, str):
                            continue
                        # check if the predicted param name is in the list of MPC_PARAM_NAMES
                        if predicted_param_name in MPC_PARAM_NAMES:
                            correct_params += 1
                        # Check if the predicted param name is a substring of any of the MPC_PARAM_NAMES
                        elif any(predicted_param_name in name for name in MPC_PARAM_NAMES) or \
                            _parse_mpc_param_name(predicted_param_name) is not None:
                            partially_correct_params += 1
                    # print(f"Num fully correct = {correct_params} num partially correct = {partially_correct_params}")
                    # print(f"Num flase param names = {len(extracted_command.keys()) - correct_params - partially_correct_params}")
                    rewards[i] = (correct_params + 0.5 * partially_correct_params) / len(extracted_command.keys())
        print(f"mpc_param_name_reward = {rewards}")
        return [rewards]


    def driving_adaption_reward(prompts, completions, evaluation_function, baseline_rmse, **kwargs) -> list[float]:
        print("############################################Driving Reward############################################")
        # Get the evaluation function of the driving style
        eval_func = getattr(eval_driving, evaluation_function[0])
        print("EVAL FUNC", evaluation_function[0])
        
        rewards = np.zeros(len(prompts))
        rmses = np.zeros(len(prompts))
        for i in range(len(prompts)):            
            # Reset MPC Params
            race_llm._reset_mpc_params()

            # wait 2 seconds for the MPC to reset
            time.sleep(2)
            # Extract the MPC Params 
            extracted_command, _ = race_llm._sanitize_tune_output(completions[i])
            if extracted_command is None:
                rmses[i] = 69.0
                continue
                
            # Set the new MPC params
            try:
                sanitized_commands = extracted_command.copy()
                if len(sanitized_commands) > 0:
                    for param in extracted_command.keys():
                        if any(param in name for name in MPC_PARAM_NAMES):
                            continue
                        else:
                            parsed_param = _parse_mpc_param_name(param)
                            if parsed_param is not None:
                                # Replace the param name with the parsed one
                                sanitized_commands[parsed_param] = sanitized_commands.pop(param)

                    
                    for param, value in sanitized_commands.items():
                        # Check if the param name is a string
                        if not isinstance(param, str):
                            print(f"Skipping param {param} because it is not a string")
                            continue
                        race_llm._set_ros_param(param, value)
                        
                print(f"Sanitized commands: {sanitized_commands}")
                # Evaluate the driving style via ROS
                time.sleep(5) # wait for the MPC to settle
                rmse = eval_func(race_llm)
                
                # Check if crashed
                crashed = race_llm._crash_detection_via_sim()
                
                # Reset the car if crashed
                if crashed:
                    rmses[i] = 69.0 # High RMSE for crashing
                    race_llm._reset_car()
                    time.sleep(5)
                else:
                    rmses[i] = rmse # RMSE for valid commands
                # print(f"RMSE: {rmse}")
                
            except Exception as e:
                # TODO: maybe not good idea to punish for something failing which is not LLM's fault
                rmses[i] = 69.0 # High RMSE for failing in setting params 
                print(f"Failed to set valid command from LLM: {completions[0]}, {str(e)}")

        # Normalize RMSEs to rewards
        rewards += _rmse_to_reward(rmses, baseline_rmse)
        
        # CHECK IF MPC CRASHED
        mpc_crashed = race_llm._mpc_crash_detection(echo_nb=200, timeout=0.5)
        if mpc_crashed:
            while mpc_crashed:
                mpc_crashed = race_llm._mpc_crash_detection(echo_nb=200, timeout=0.5)
                print(f"MPC CRASHED, WAITING FOR RESTARTING OF SIM")
                time.sleep(5)
            print("MPC IS BACK!!!!!!!!!!")
        
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
            
        print(f"Driving reward: {rewards}, RMSEs: {rmses}")
        print("############################################Driving Reward End############################################")
        return [rewards]
    
    def _parse_mpc_param_name(predicted_param_name: str) -> str:
        # Define a mapping of common variations to standard names
        param_mapping = {
            "qv": ["q_v", "weight_qv", "param_qv", "weight_q_v", "param_q_v"],
            "qn": ["q_n", "weight_qn", "param_qn", "weight_q_n", "param_q_n"],
            "qalpha": ["q_alpha", "weight_qalpha", "param_qalpha", "weight_q_alpha", "param_q_alpha"],
            "qac": ["q_ac", "weight_qac", "param_qac", "weight_q_ac", "param_q_ac"],
            "qddelta": ["q_ddelta", "weight_qddelta", "param_qddelta", "weight_q_ddelta", "param_q_ddelta"],
            "alat_max": ["a_lat_max", "weight_alat_max", "param_alat_max", "weight_a_lat_max", "param_a_lat_max"],
            "a_min": ["a_min", "weight_a_min", "param_a_min"],
            "a_max": ["a_max", "weight_a_max", "param_a_max"],
            "v_min": ["v_min", "weight_v_min", "param_v_min"],
            "v_max": ["v_max", "weight_v_max", "param_v_max"],
            "track_safety_margin": ["track_safety_margin", "weight_track_safety_margin", "param_track_safety_margin"],
        }
        # Return the standardized name or the original if not found
        for standard_name, variations in param_mapping.items():
            if predicted_param_name in variations:
                # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                # print(f"Predicted param name '{predicted_param_name}' mapped to standard name '{standard_name}'")
                # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                return standard_name
        return None
    
    def _rmse_to_reward(rmses: np.ndarray, baseline_rmse: float, min_out: float = 0.0) -> np.ndarray:
        if rmses.size == 0:
            return np.ones_like(rmses, dtype=float) * min_out
        
        rel_improvement = (baseline_rmse - rmses) / baseline_rmse
        # print(f"Relative improvement: {rel_improvement}, baseline RMSE: {baseline_rmse}, RMSEs: {rmses}")

        # clip to [-4, 4]
        rel_improvement = np.clip(rel_improvement, -4, 4)

        return rel_improvement


    # Custom wandb callback
    class WandbCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and use_wandb:
                wandb.log(logs)

    class StepTrackerCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            step_state["step"] = state.global_step

    dataset = MPCDatasetGRPO(
        test_cases=eval_driving.DRIVING_CASES,
        use_rag=use_rag,
        index=race_llm.decision_index,
        mem_nb=cfg["dataset"]["mem_nb"],
        shuffle=cfg["dataset"]["shuffle"],
    )

    print(f"Training MPCxLLM with {len(dataset)} samples.")

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
        
        if cfg["evaluation"]["evaluation_on"]: 
            # Add custom callbacks if enabled
            eval_fn = partial(evaluate_model, chat_template=chat_template, run_name=run_name)
            assert cfg["grpo"]["max_steps"] % cfg["evaluation"]["eval_interval_steps"] == 0, "max_steps must be divisible by eval_interval_steps"
            callbacks = [
                WandbCallback(),
                PeriodicEvalCallback(eval_fn=eval_fn, shared_ros=shared_ros, model=model, tokenizer=tokenizer,
                                    host_ip=cfg["training"]["ros_host_ip"], interval=cfg["evaluation"]["eval_interval_steps"],
                                    run_name=run_name),
                StepTrackerCallback(),
            ]
        else:
            print("No evaluation on training, using default callbacks")
            callbacks = [
                WandbCallback(),
                StepTrackerCallback(),
            ]

        trainer = GRPOTrainer(
            model = model,
            processing_class = tokenizer,
            reward_funcs = [
                wrapped_driving_reward,
                wrapped_format_reward,
                mpc_param_extraction_reward,
                mpc_param_name_reward
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

def evaluate_model(model, shared_ros, tokenizer, host_ip, chat_template="qwen-2.5", step=0,
                   run_name="no_wandb"):
    print("#########################################################")
    print(f"DOING EVAL WITH HOST IP: {host_ip}")
    
    tester = MPCTester(
        openai_token=OPENAI_API_TOKEN,
        custom_model=model,
        shared_ros=shared_ros,
        run_name=run_name,
        step=step,
        custom_tokenizer=tokenizer,
        custom_chat_template=chat_template,
        host_ip=host_ip
    ) 

    # Run tests and plot results
    reports = tester.run_tests(num_tests=None, num_memories=5)

    # Log logfile to wandb
    if wandb.run is not None:
        for test_case, logs in reports.items():
            for sub_case in logs["subcases"]:
                logfile = logs["subcases"][sub_case]["logfile"]
                artifact = wandb.Artifact(name=f"mpc_eval_log_{test_case}_{sub_case}_step{step}", type="mpc_evaluation")
                artifact.add_file(logfile)
                wandb.run.log_artifact(artifact)
            # Log results to wandb
            wandb.log({f"eval/mpc_eval_{test_case}_avg_rmse": logs["avg_rmse"]})
            wandb.log({f"eval/mpc_eval_{test_case}_avg_improvement": logs["avg_rmse_improvement"]})
            
        
    print("#########################################################")
    return 0
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, 
                        default="train/config/rl_mpc_train.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    train(load_config(args.config))