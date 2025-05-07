import os
import re
import random
import json
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from llm_mpc import RaceLLMMPC
from typing import List

# Load environment variables
load_dotenv(find_dotenv())
OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize the LLMs
gpt = ChatOpenAI(model_name='o1-preview', openai_api_key=OPENAI_API_TOKEN)
race_llm = RaceLLMMPC(hf_token=HF_API_TOKEN, openai_token=OPENAI_API_TOKEN, model='o1-preview', no_ROS=False)

# Function to construct the LLM query prompt for MPC tuning
def build_mpc_prompt(race_llm : RaceLLMMPC, scenario : str, memory_nb : int=0) -> str:
    # Generate the LLM MPC query
    LLM_query = race_llm.race_mpc_interact(scenario=scenario, memory_nb=memory_nb, prompt_only=True)
    return LLM_query

def build_reasoning_prompt(race_llm : RaceLLMMPC) -> str:
    # Generate the LLM Reasoning query
    LLM_query = race_llm.race_reasoning(prompt_only=True)
    # Augment the query to explain the reasoning
    pattern = r"Instruct the car to take the appropriate action very briefly, no explanation needed:\s*I want the car to prioritize: <Action>!"
    replace_str = "Instruct the car to take the appropriate action very briefly and explain why you chose the action:\nI want the car to prioritize: <Action>! Because of <Explanation>."
    
    # Perform the replacement
    LLM_query = re.sub(pattern, replace_str, LLM_query)
    return LLM_query

# Function to construct a conversation object
def construct_conversation(human_prompt: str, gpt_response: str) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": human_prompt},
            {"from": "gpt", "value": gpt_response}
        ]
    }

# Function to load an existing dataset
def load_existing_dataset(filename):
    if os.path.exists(filename):
        if os.path.getsize(filename) > 0:  # Check if the file is not empty
            with open(filename, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: {filename} is corrupted. Starting with an empty dataset.")
                    return []  # Return empty list if JSON is corrupted
        else:
            print(f"Warning: {filename} is empty. Starting with an empty dataset.")
            return []  # Return empty list if file is empty
    return []

# Function to save a conversation to the dataset
def save_conversation_to_file(filename: str, conversation: dict):
    dataset = load_existing_dataset(filename)
    dataset.append(conversation)
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=4)


# Main dataset generation loop
filename = 'train/dataset/combined_reason_mpc_dataset.json'
for i in range(200):
    reasoning_query = build_reasoning_prompt(race_llm=race_llm)
    reasoning_llm_out = gpt.invoke(reasoning_query, temperature=1.0).content
    # Apply LLMxMPC to get new kinds of data
    _, _, _, mpc_query, raw_mpc_llm_out = race_llm.race_mpc_interact(scenario=reasoning_llm_out, memory_nb=random.randint(0, 8))

    combined_query = f"{reasoning_query}\n{mpc_query}"
    combined_llm_out = f"{reasoning_llm_out}\n{raw_mpc_llm_out}"

    # Construct the conversation
    conversation = construct_conversation(human_prompt=combined_query, gpt_response=combined_llm_out)

    # Save the conversation to the file
    save_conversation_to_file(filename, conversation)
    print('-----------------------------------')