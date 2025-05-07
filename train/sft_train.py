import argparse, yaml
from datetime import datetime
from unsloth import FastLanguageModel, save
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from peft.peft_model import PeftModelForCausalLM
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from dotenv import load_dotenv, find_dotenv
import os, json
from typing import Any, Dict

load_dotenv(find_dotenv())

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def preprocess(text, chat_template="phi-3", answer=False):
    if chat_template == "phi-3":
        if answer:
            messages = [{"from": "gpt", "value": text}]
        else:
            messages = [{"from": "human", "value": text}]
    elif chat_template == "qwen-2.5":
        if answer:
            messages = [{"role": "assistant", "content": text}]
        else:
            messages = [{"role": "user", "content": text}]
    else:
        raise ValueError(f"Chat template {chat_template} not recognized. Please use 'phi-3' or 'qwen-2.5'.")
    return messages

def chat_mapping(chat_template="phi-3"):
    if chat_template == "phi-3":
        return {"role": "from", "content": "value", "user": "human", "assistant": "gpt"}
    elif chat_template == "qwen-2.5":
        return {"role": "role", "content": "content", "user": "user", "assistant": "assistant"}
    else:
        raise ValueError(f"Chat template {chat_template} not recognized. Please use 'phi-3' or 'qwen-2.5'.")

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train(cfg: Dict[str, Dict[str, Any]]):
    out_dir: str = cfg["training"]["out_dir"]
    chat_template: str = cfg["training"]["chat_template"]
    seed: int = cfg["training"]["seed"]

    base_model: str = cfg["model"]["base_model"]
    experiment_name: str = base_model.split("/")[-1] + "_SFT_"
    experiment_name += datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = cfg["model"]["base_model"],
        max_seq_length = cfg["model"]["max_seq_length"],
        dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = cfg["model"]["load_in_4bit"],
        token=HUGGINGFACEHUB_API_TOKEN
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = cfg["model"]["lora_rank"],
        target_modules = cfg["model"]["target_modules"],
        lora_alpha = cfg["model"]["lora_alpha"],
        lora_dropout = cfg["model"]["lora_dropout"],
        bias = "none",
        use_gradient_checkpointing = cfg["model"]["use_gradient_checkpointing"],
        random_state = seed,
        use_rslora = False,
        loftq_config = None,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template,
        mapping=chat_mapping(),
    )

    ####################################CUSTOM DATA####################################
    def formatting_prompts_custom_func(examples):
        convos = []
        for i in range(len(examples["conversations"])):
            input_text = examples["conversations"][i][0]["value"]
            output_text = examples["conversations"][i][1]["value"]
            input_sample = preprocess(text=input_text, chat_template=chat_template, answer=False)
            output_sample = preprocess(text=output_text, chat_template=chat_template, answer=True)
            sample = input_sample + output_sample
            convos.append(sample)
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return { "text" : texts, }

    # Create a combined json file from all json files in the dataset folder
    dataset_dir = os.path.join(os.getcwd(), cfg["training"]["dataset_dir"])
    all_json_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.json')]
    # Initialize an empty list to hold all conversations
    all_conversations = []
    # Read and combine all JSON files
    for fname in all_json_files:
        print(f"Reading: {fname}")
        with open(fname) as infile:
            data = json.load(infile)
            all_conversations.extend(data)
    # Write the combined data to a new JSON file
    combined_json_path = os.path.join(dataset_dir, 'combined/full_data.json')
    with open(combined_json_path, 'w') as outfile:
        json.dump(all_conversations, outfile, indent=4)

    custom_dataset = load_dataset('json', data_files=combined_json_path, split='train')
    dataset = custom_dataset.map(formatting_prompts_custom_func, batched=True)
    ####################################TRAINING####################################
    if cfg["training"]["train_bool"]:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=cfg["model"]["max_seq_length"],
            dataset_num_proc=2,
            packing=False, # Can make training 5x faster for short sequences.
            args=TrainingArguments(
                per_device_train_batch_size=cfg["trainer"]["per_device_train_batch_size"],
                gradient_accumulation_steps=cfg["trainer"]["gradient_accumulation_steps"],
                warmup_steps=cfg["trainer"]["warmup_steps"],
                max_steps=cfg["trainer"]["max_steps"],
                learning_rate=cfg["trainer"]["learning_rate"],
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=cfg["trainer"]["logging_steps"],
                optim=cfg["trainer"]["optim"],
                weight_decay=cfg["trainer"]["weight_decay"],
                lr_scheduler_type=cfg["trainer"]["lr_scheduler_type"],
                seed=seed,
                output_dir=out_dir,
            ),
        )

        # Stats
        trainer_stats = trainer.train()
        print("Training stats: ", trainer_stats)

    model.save_pretrained(f"{out_dir}/{experiment_name}")
    print("Training complete, saved model: ", f"{out_dir}/{experiment_name}")
    
    if cfg["training"]["create_merged_model"]:
        create_merged(model, tokenizer, out_dir=out_dir)
        print("Merged model saved to: ", f"{out_dir}/merged")
    
    return model, tokenizer

def create_merged(model: PeftModelForCausalLM, tokenizer: LlamaTokenizerFast, out_dir="outputs/race_llm"):
    merged_dir = os.path.join(out_dir, "merged")

    # Create merged directory if not existing
    if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)

    save.unsloth_save_model(model=model, 
                            tokenizer=tokenizer, 
                            save_directory=merged_dir,
                            push_to_hub=False,
                            save_method="merged_16bit",)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, 
                        default="train/config/sft_train.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    train(load_config(args.config))
