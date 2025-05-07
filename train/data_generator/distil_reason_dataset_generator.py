import os, re, random, json
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from llm_mpc import RaceLLMMPC

load_dotenv(find_dotenv())
OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

gpt4o = ChatOpenAI(model_name='gpt-4o', openai_api_key=OPENAI_API_TOKEN)
gpt4omini = ChatOpenAI(model_name='gpt-4o-mini', openai_api_key=OPENAI_API_TOKEN)

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

def construct_conversation(human_prompt, gpt_response):
    return {
        "conversations": [
            {
                "from": "human",
                "value": human_prompt
            },
            {
                "from": "gpt",
                "value": gpt_response
            }
        ]
    }

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

def save_conversation_to_file(filename, conversation):
    # Load existing data
    dataset = load_existing_dataset(filename)
    
    # Append the new conversation
    dataset.append(conversation)
    
    # Write updated dataset to file
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=4)

def distil_reasoning_from_ros_mpc(filename='train/dataset/gpt_distil_reason_dataset.json'):
    race_llm = RaceLLMMPC(hf_token=HF_API_TOKEN, openai_token=OPENAI_API_TOKEN, model='gpt-4o', no_ROS=False)

    for i in range(200):
        reasoning_query = build_reasoning_prompt(race_llm=race_llm)
        reasoning_llm_out = gpt4o.invoke(reasoning_query).content
        print('Reasoning_query', reasoning_query, 'Reasoning_llm_out', reasoning_llm_out)
        # Apply LLMxMPC to get new kinds of data
        race_llm.race_mpc_interact(scenario=reasoning_llm_out, memory_nb=random.randint(0, 8))

        if 'Reverse' in reasoning_llm_out:
            # Construct the conversation
            conversation = construct_conversation(human_prompt=reasoning_query, gpt_response=reasoning_llm_out)

            # Save the conversation to the file
            save_conversation_to_file(filename, conversation)
        print('-----------------------------------')

def distil_reasoning_from_dataset(dataset='train/dataset/synthetic_robot_data.json', out_filename='train/dataset/gpt_distil_reasonfromdata_dataset.json'):
    dataset = load_existing_dataset(dataset)
    dataset = dataset[-30:]  # Only use the last 30 conversations
    for conversation in dataset:
        reasoning_query = conversation['conversations'][1]['value']
        llm_prompt = f"""Analyze the provided data from an autonomous racing car driving on a track in the Frenet coordinate frame. 
        Focus on identifying the car's current state by referring to the provided data (e.g., position, speed, distance to walls, and any safety concerns) 
        and suggest the most appropriate action, with a brief explanation of why by 
        reasoning on the provided state:
        - State towards which side of the wall the car is closer to and how close it is.
        - State the average s-velocity of the car and relate if this is fast, slow or nominal.
        - State the average d-velocity of the car and relate if this is fast, slow or nominal.
        - State if the car is facing the wall.
        - State if the car has crashed.

        Also be aware that reversing is a must if the car has crashed and is facing the wall!!!!

        Then based on this explain why you chose the action you did.\n\n {reasoning_query}\n"""
        reasoning_llm_out = gpt4o.invoke(llm_prompt).content
        rephrased_reasoning_query = gpt4omini.invoke(f"Slightly rephrase the prompt but leave the values as they are: {reasoning_query}").content

        conversation = construct_conversation(human_prompt=rephrased_reasoning_query, gpt_response=reasoning_llm_out)
        save_conversation_to_file(out_filename, conversation)
        print(conversation)
        print('-----------------------------------')

if __name__ == '__main__':
    #distil_reasoning_from_ros_mpc()
    distil_reasoning_from_dataset()