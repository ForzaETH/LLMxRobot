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

def load_prompt(prompt_type) -> str:
        if 'reasoning' in prompt_type:
            hints_dir = os.path.join('./', 'prompts/reasoning_hints.txt')
            with open(hints_dir, 'r') as f:
                reasoning_hints = f.read()
            return reasoning_hints
        elif 'synthetic' in prompt_type:
            hints_dir = os.path.join('./', 'prompts/example_synthetic.txt')
            with open(hints_dir, 'r') as f:
                synthetic_hints = f.read()
            return synthetic_hints
        else:
            raise ValueError(f"Prompt type {prompt_type} not recognized. Please use 'reasoning' or 'synthetic'.")

def generate_synthetic_race_data(filename='train/dataset/synthetic_robot_data.json', driving_style="Centered on the centerline without crashes"):
    reasoning_hints = load_prompt(prompt_type='reasoning')
    synth_example = load_prompt(prompt_type='synthetic')
    for i in range(25):
        data_time = random.randint(1, 5)
        data_samples = random.randint(max(3, data_time), 15)
        prompt = f"""
        I want you to generate synthetic data for a robot racing car. The car is currently driving on a track, data available is in the Frenet Coordinate frame, units are in meters and meters per second. 
        The racing line is a minimal curvature trajectory to optimize the lap time, which is not the centerline of the track.

        The data has to represent the car driving: {driving_style}.
        
        Here are some hints of nominal driving behavior: \n{reasoning_hints}\n\n

        The crash_bool and the facing_wall bool are single values, while the rest of the data is sampled for {data_time} second in {data_samples} samples and represented as list of {data_samples} entries.

        Here an example output: \n{synth_example}\n

        STRICTLY ADHERE TO THIS TEMPLATE AND OUTPUT IT AS TEXT:
        The data has been sampled for {data_time} seconds in {data_samples} samples.
        - The car's position along the racing line is given by the s-coordinate: s_pos\n
        - The car's lateral deviation from the racing line is given by the d-coordinate: d_pos\n
        - The car's speed along the racing line is given by the s-speed: s_speed\n
        - The car's speed perpendicular to the racing line is given by the d-speed: d_speed\n
        - The distance to the left wall is: d_left\n
        - The distance to the right wall is: d_right\n
        - Bool if the car has crashed: crashed_bool\n
        - Bool if the car is facing the wall: facing_wall\n
        Explanation: Short analysis why the data represents the car driving: {driving_style}\n
        """
        synth_data = gpt4o.invoke(prompt).content

        human_base_prompt = f"Generate synthetic data for a robot racing car driving on a track in the Frenet coordinate frame. The data has to represent the car driving: {driving_style}."
        human_prompt = gpt4o.invoke(f"Rephrase this sentence: {human_base_prompt}").content
        conversation = construct_conversation(human_prompt=human_prompt, gpt_response=synth_data)
        save_conversation_to_file(filename, conversation)
        print(conversation)
        print('-----------------------------------')

def generate_dummy_math_logic(filename='train/dataset/dummy_math.json'):
    dumb_boi = "Yes, 0.74 is within the range of 5-7. To explain why, we need to understand that the question seems to be asking whether the number 0.74 falls between the numbers 5 and 7 on a number line. Since 0.74 is less than 1 and both 5 and 7 are greater than 1, it is clear that 0.74 is not between 5 and 7. The number 0.74 is actually closer to 0 than to 5 or 7 on the number line."
    for i in range(10):
        prompt = f"I have a small LLM that struggles with the concept of determining if a value is within a given range. Look at it's output: {dumb_boi}. Give me a nice question for it to solve and randomize the values."
        dummy_math = gpt4omini.invoke(prompt).content
        smart_math = gpt4o.invoke(dummy_math).content

        conversation = construct_conversation(human_prompt=dummy_math, gpt_response=smart_math)
        save_conversation_to_file(filename, conversation)
        print(conversation)
        print('-----------------------------------')

if __name__ == '__main__':
    driving_style = input("The data has to represent the car driving...")
    generate_synthetic_race_data(driving_style=driving_style)

    #generate_dummy_math_logic()