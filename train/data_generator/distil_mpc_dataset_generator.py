import os, re, random, json
from dotenv import load_dotenv, find_dotenv
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from llm_mpc import RaceLLMMPC

load_dotenv(find_dotenv())
OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

gpt4o = ChatOpenAI(model_name='gpt-4o', openai_api_key=OPENAI_API_TOKEN)
race_llm = RaceLLMMPC(hf_token=HF_API_TOKEN, openai_token=OPENAI_API_TOKEN, model='gpt-4o', no_ROS=True)

def randomize_parameters(input_str):
    # Regex pattern to find the min, max, and default values
    pattern = r'(\w+)\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)'
    
    def randomize_match(match):
        param_name = match.group(1)
        min_val = float(match.group(2))
        max_val = float(match.group(3))

        # For v_min and v_max, ensure they allow for negative velocities
        if param_name in ["v_min", "v_max"]:
            # Randomize min and max, ensuring min < max
            new_min = round(random.uniform(min_val, 0), 1)
            new_max = round(random.uniform(new_min, max_val), 1)
        
        else:
            # Randomize min and max, ensuring min < max
            new_min = round(random.uniform(min_val, max_val), 1)
            new_max = round(random.uniform(new_min, max_val), 1)
        
        # Randomize default, ensuring it is between new_min and new_max
        new_default = round(random.uniform(new_min, new_max), 1)
        
        return f"{param_name} {new_min}, {new_max}, {new_default}"
    
    # Substitute using the randomize_match function
    randomized_str = re.sub(pattern, randomize_match, input_str)
    
    return randomized_str

def build_prompt(race_llm : RaceLLMMPC, base_memory : str, scenario : str, memory_nb : int=0) -> str:
    # Query the index and pass the result to the command chain for processing
        RAG_query = f"""
        Task: {scenario}\n
        """
        #Perform RAG manually
        #Retrieve docs from the RAG
        rag_sources: List[Document] = race_llm.vector_index.vectorstore.search(query=RAG_query, search_type='similarity', k=memory_nb) if memory_nb > 0 else []
        rag_sources = [{'meta': doc.metadata, 'content': doc.page_content} for doc in rag_sources]
        LLM_query = f"""
        You are an AI assistant helping to tune the parameters of an MPC controller for an autonomous racing car. Below is the context and the task:

        ## Context
        1. **Scenario**: {scenario}
        2. **Base Memory and Cost Formulation**: {base_memory}

        ## Task
        Adapt the tuneable parameters of the MPC so that the car achieves the following: **{scenario}**.

        ## Constraints
        - Use the min and max ranges of the parameters provided in the base memory.
        - Focus on the primary task and utilize the chat history if necessary.
        - Only consider relevant RAG information and quote it briefly in your explanation.

        ## RAG Information
        - !!! Not all memories are relevant to the task. Select the most relevant ones. !!!
        - **Memory Entries**:
        {rag_sources}

        ## Expected Output Format
        Always strictly return your answers in the following format (no other dicts in your response and no comments in the params part!):
        new_mpc_params = {{
            'param1': new_value1,
            'param2': new_value2,
            ...
        }}

        Explanation: <Your very short explanation here>
        """
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

filename = 'train/dataset/gpt_distil_mpc_dataset.json'
scenarios = [
     "Complete a lap around the track in the shortest time possible",
     "Drive as safely as possible around the track",
     "Stay away from the walls while driving around the track",
     "Reverse the car around the track",
     "Do not exceed speeds of 10 km/h while driving around the track",
     "Drive around the track without using the brakes",
]
for i in range(len(scenarios)):
    scenario = scenarios[i]
    for j in range(3):
        base_mem = randomize_parameters(input_str=race_llm.base_memory)
        last_rephrase = scenario
        for k in range(10):
            rephrased_scenario = gpt4o.invoke(f'Rephrase this:\n {last_rephrase}').content
            last_rephrase = rephrased_scenario
            llm_query = build_prompt(race_llm=race_llm, base_memory=base_mem, scenario=rephrased_scenario, memory_nb=k)
            print('LLM_query', llm_query)
            llm_out = gpt4o.invoke(llm_query).content
            print('LLM_out', llm_out)
            # Construct the conversation

            conversation = construct_conversation(human_prompt=llm_query, gpt_response=llm_out)
            
            # Save the conversation to the file
            save_conversation_to_file(filename, conversation)
    print('-----------------------------------')
