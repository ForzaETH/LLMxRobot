import os, re, random, json
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from llm_mpc import RaceLLMMPC

load_dotenv(find_dotenv())
OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

gpt4o = ChatOpenAI(model_name='gpt-4o', openai_api_key=OPENAI_API_TOKEN)
gpt4omini = ChatOpenAI(model_name='gpt-4o-mini', openai_api_key=OPENAI_API_TOKEN)

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

# Loads Prompt with hints
def load_prompt(prompt_type) -> str:
    print('Loading prompt from CWD:', os.getcwd())
    if 'reasoning' in prompt_type:
        hints_dir = os.path.join('.', 'prompts/reasoning_hints.txt')
        with open(hints_dir, 'r') as f:
            reasoning_hints = f.read()
        return reasoning_hints
    elif 'synthetic' in prompt_type:
        hints_dir = os.path.join('.', 'prompts/example_synthetic.txt')
        with open(hints_dir, 'r') as f:
            synthetic_hints = f.read()
        return synthetic_hints
    else:
        raise ValueError(f"Prompt type {prompt_type} not recognized. Please use 'reasoning' or 'synthetic'.")

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

def randomize_text_parameters(input_str, lower_bound=0.6, upper_bound=1.8):
    # Pattern to match numbers followed by 'm' or 'm/s'
    pattern = r'(\d+(?:\.\d+)?)(m/s|m)'

    rand_frac = random.uniform(lower_bound, upper_bound)
    
    def randomize_match(match):
        original_value = float(match.group(1))
        unit = match.group(2)
        
        # Apply % randomization
        new_value = round(original_value * rand_frac, 2)
        
        return f"{new_value}{unit}"
    
    # Replace numerical values in the text with randomized ones
    randomized_str = re.sub(pattern, randomize_match, input_str)
    return randomized_str

def build_analysis_prompt(robot_state : str, randomized_hints : str) -> str:
    prompt = f"""
    {robot_state}\n
    Here are some hints to help you analyze the state: {randomized_hints}\n
    
    Focus on briefly identifying the car's current state by referring to the provided data (e.g., position, speed, distance to walls, and any safety concerns).
    - State towards which side of the wall the car is closer to and if this is too close to the wall or not.
    - State the average s-velocity of the car and relate if this is fast, slow or nominal.
    - State the average d-velocity of the car and relate if this is fast, slow or nominal.
    - State the magnitude of the positional oscillation of the car and if they are nominal or too high.
    - State the magnitude of the velocity oscillation of the car and if they are nominal or too high.
    - State if the car is facing the wall.
    - State if the car has crashed.
    - State if the car is reversing, driving forward, or stopped.
    """
    return prompt

def build_decision_prompt(human_prompt : str, robot_state : str, randomized_hints : str) -> str:
    # Query the index and pass the result to the command chain for processing
    prompt = f"""
    The human wants the car to drive: {human_prompt}\n
    Check if the car is adhering to the human's command or not.\n

    {robot_state}\n
    Here are some hints to help you analyze the state: {randomized_hints}\n
    
    Briefly identify the car's current state by referring to the provided data (e.g., position, speed, distance to walls, and any safety concerns from the hints).

    Decide if the car is adhering to the human's desired drving style of: {human_prompt} by choosing from the two actions:\n

    a) Continue: The car is driving as expected and should continue driving in the same manner.
    b) Correct: The car is not driving as expected and state how the car should correct its driving style.

    Strictly adhere to the reply format:

    State Recap: <Brief Explanation>
    Action: <a or b>

    """
    return prompt

def distil_randomized_state_analysis(dataset='train/dataset/gpt_distil_reason_dataset.json', out_filename='train/dataset/randomized_state_analysis.json'):
    og_hints = load_prompt('reasoning')
    dataset = load_existing_dataset(dataset)
    for conversation in dataset:
        state_with_query = conversation['conversations'][0]['value']
        robot_state = state_with_query.split("\n\n\n        Here are some hints to help you reason about the car's current state:")[0]
        randomized_hints = randomize_text_parameters(og_hints)

        llm_prompt = build_analysis_prompt(robot_state, randomized_hints)
       
        state_analyze = gpt4o.invoke(llm_prompt).content

        conversation = construct_conversation(human_prompt=llm_prompt, gpt_response=state_analyze)
        save_conversation_to_file(out_filename, conversation)
        print(conversation)
        print('-----------------------------------')

def distil_randomized_decision_making(dataset='train/dataset/excluded/gpt_distil_reason_dataset.json', out_filename='train/dataset/randomized_decision_making.json'):
    og_hints = load_prompt('reasoning')
    dataset = load_existing_dataset(dataset)

    # scenarios = [
    #     "Stop the car!",
    #     "Drive as safely as possible around the track",
    #     "Stay away from the walls while driving around the track",
    #     "Reverse the car around the track",
    #     "Do not exceed speeds of 4m/s while driving around the track",
    #     "I want the car to drive as smoothly as possible",
    # ]
    scenarios = [
        "Drive at speeds above 4.75m/s!",
        "Drive at speeds between 2.5m/s and 5.5m/s!",
    ]

    for conversation in dataset:
        state_with_query = conversation['conversations'][0]['value']
        robot_state = state_with_query.split("\n\n\n        Here are some hints to help you reason about the car's current state:")[0]

        for scenario in scenarios:
            randomized_hints = randomize_text_parameters(og_hints)

            # Randomize number values in the scenario
            if 'm/s' in scenario:
                scenario = randomize_text_parameters(scenario, lower_bound=0.8, upper_bound=1.9)

            rephrased_human_prompt = gpt4omini.invoke(f'Rephrase this:\n {scenario}').content

            llm_prompt = build_decision_prompt(rephrased_human_prompt, robot_state, randomized_hints)

            decision_making = gpt4o.invoke(llm_prompt).content

            conversation = construct_conversation(human_prompt=llm_prompt, gpt_response=decision_making)
            save_conversation_to_file(out_filename, conversation)
            print(conversation)
            print('-----------------------------------')


if __name__ == '__main__':
    #distil_randomized_state_analysis()
    distil_randomized_decision_making()