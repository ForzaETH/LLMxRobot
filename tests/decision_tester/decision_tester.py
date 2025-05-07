from dotenv import load_dotenv, find_dotenv
import os, json, datetime
import numpy as np
import tqdm
import argparse
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAIEmbeddings

class DecisionTester:
    def __init__(self, llm, model_name, all_tests=False, mini=False, local=True, use_rag=False, quant=False):
        self.llm = llm
        self.model_name = model_name.replace("/", "_")
        self.all_tests = all_tests
        self.mini_eval = mini
        self.local_inference = local
        self.use_rag = use_rag
        self.quant = quant
        if self.use_rag:
            # Get Memories for the RAG
            memories_dir = 'prompts/RAG_memory.txt'
            print(f'Loading memories from {memories_dir}...')
            memories_loader = TextLoader(file_path=memories_dir)

            # Create a VectorstoreIndex from the collected loaders
            self.splitter = CharacterTextSplitter(separator='#', keep_separator=False, chunk_overlap=20, chunk_size=100)
            self.index = VectorstoreIndexCreator(embedding=OpenAIEmbeddings(api_key=OPENAI_API_TOKEN), text_splitter=self.splitter).from_loaders([memories_loader])
            self.mem_nb = 5


        if self.mini_eval:
            print("RUNNING MINI TEST")

        # Ensure the logs directory exists
        if all_tests:
            self.full_or_mini = "full" if not self.mini_eval else "mini"
            self.logs_dir = f"tests/decision_tester/logs/{self.model_name}_{self.full_or_mini}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        else:
            self.logs_dir = "tests/decision_tester/logs/"
        print(f"Logging to {self.logs_dir}")
        # Create the logs directory first time only
        os.makedirs(self.logs_dir, exist_ok=True)

        # Define test cases
        self.TEST_CASES =  [
            {"human_prompt":"Don't move", "evaluation_function": self._stop_car_check},
            {"human_prompt":"Reverse the car", "evaluation_function": self._reverse_car_check},
            {"human_prompt":"Drive forward", "evaluation_function": self._forward_check},
            {"human_prompt":"Oscillate!", "evaluation_function": self._oscillation_check},
            {"human_prompt":"Drive close to the left wall", "evaluation_function": self._drive_close_to_left_wall_check},
            {"human_prompt":"Drive on the centerline", "evaluation_function": self._drive_centerline_check},
            {"human_prompt":"Drive faster than 3 m/s", "evaluation_function": self._speed_check},
            {"human_prompt":"Drive on the racing line", "evaluation_function": self._drive_racing_line_check},
        ]


    #################Checks#################
    def _stop_car_check(self, robot_states):
        evaluation_list = []
        for i in robot_states:
            if np.abs(np.mean(i["s_speed"])) < 0.1 and np.abs(np.mean(i["d_speed"])) < 0.1:
                evaluation_list.append(True)
            else:
                evaluation_list.append(False)
        return evaluation_list

    def _reverse_car_check(self, robot_states):
        evaluation_list = []
        for i in robot_states:
            if np.mean(i["s_speed"]) < 0 or i["reversing"] == True:
                evaluation_list.append(True)
            else:
                evaluation_list.append(False)
        return evaluation_list

    def _forward_check(self, robot_states):
        evaluation_list = []
        for i in robot_states:
            if np.mean(i["s_speed"]) > 0.1:
                evaluation_list.append(True)
            else:
                evaluation_list.append(False)
        return evaluation_list

    def _oscillation_check(self, robot_states):
        evaluation_list = []
        for i in robot_states:
            sign_changes = 0
            for j in range(1, len(i["d_pos"])):
                if i["d_pos"][j] * i["d_pos"][j-1]  < 0:  # Sign change occurs
                    sign_changes += 1
            if sign_changes > 1 and np.abs(max(i["d_pos"])) >= 0.3:
                evaluation_list.append(True)
            else:
                evaluation_list.append(False)
        return evaluation_list

    def _drive_close_to_left_wall_check(self, robot_states):
        evaluation_list = []
        for i in robot_states:
            if np.mean(i["d_left"]) < 0.4 and np.mean(i["s_speed"]) > 0.1:
                evaluation_list.append(True)
            else:
                evaluation_list.append(False)
        return evaluation_list

    def _drive_centerline_check(self, robot_states):
        evaluation_list = []
        for i in robot_states: 
            on_centerline = True
            for j in range(i["data_samples"]):
                if np.abs(i["d_left"][j] - i["d_right"][j]) > 0.3 and np.mean(i["s_speed"]) > 0.1:
                    on_centerline = False
            evaluation_list.append(on_centerline)
        return evaluation_list

    def _speed_check(self, robot_states, threshold=3):
        evaluation_list = []
        for i in robot_states:
            if np.mean(i["s_speed"]) > threshold:
                evaluation_list.append(True)
            else:
                evaluation_list.append(False)
        return evaluation_list

    def _drive_racing_line_check(self, robot_states):
        evaluation_list = []
        for i in robot_states:
            if np.abs(np.mean(i["d_pos"])) <= 0.3 and np.mean(i["s_speed"]) > 0.1:    
                evaluation_list.append(True)
            else:
                evaluation_list.append(False)
        return evaluation_list
    #################Checks#################

    def load_dataset(self, data_dir: str) -> json:
        with open(file=data_dir, mode='r') as f:
            data = json.load(f)
        return data

    def build_prompt(self, human_prompt, robot_state) -> str:
        # Hints are empty if not using RAG
        hints = ''
        if self.use_rag:
            rag_sources = self.index.vectorstore.search(query=human_prompt, search_type='similarity', k=self.mem_nb) if self.mem_nb > 0 else []
            rag_sources = [{'meta': doc.metadata, 'content': doc.page_content} for doc in rag_sources]
            for hint in rag_sources:
                hints += hint['content'] + "\n"

        prompt = f"""
        You are an AI embodied on an autonomous racing car. The human wants to: {human_prompt} \n
        The car is currently on the track, data available is in the Frenet Corrdinate frame, units are in meters and meters per second. 
        The racing line is a minimal curvature trajectory to optimize the lap time.
        The data has been sampled for {robot_state["time"]} seconds in {robot_state["data_samples"]} samples.\n        
        - The car's position along the racing line is given by the s-coordinate: {robot_state["s_pos"]}\n\n        
        - The car's lateral deviation from the racing line is given by the d-coordinate: {robot_state["d_pos"]}\n\n        
        - The car's speed along the racing line is given by the s-speed: {robot_state["s_speed"]}\n\n        
        - The car's speed perpendicular to the racing line is given by the d-speed: {robot_state["d_speed"]}\n\n        
        - The distance to the left wall is: {robot_state["d_left"]}\n\n
        - The distance to the right wall is: {robot_state["d_right"]}\n\n 
        - Bool if the car is reversing: {robot_state["reversing"]}\n\n          
        - Bool if the car has crashed: {robot_state["crashed"]}\n\n        
        - Bool if the car is facing the wall: {robot_state["facing_wall"]}\n\n\n   
        Use these guides to reason: \n\n{hints}\n\n    
        Check if the car is adhering to what the human wants: {human_prompt}. Strictly reply in the following format: \n
        Explanation: <Brief Explanation> \n
        Adhering to Human: <True/False> \n
        """
        return prompt

    def sanitize_output(self, output):
        output = next((line.strip().lower() for line in output.splitlines() if "Adhering to" in line), None)
        if 'true' in output:
            return True
        elif 'false' in output:
            return False
        else:
            return None

    def eval_decision_making(self, data_dir, llm, data_name):
        # Load dataset
        print(f" Evaluating decision making on {data_name}")
        data_set = self.load_dataset(data_dir)

        log_file = os.path.join(self.logs_dir, f"test_log_{self.model_name}_{data_name}.txt") if self.all_tests else os.path.join(self.logs_dir, f"test_log_{self.model_name}_{self.full_or_mini}_{data_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")

        # downsample the data set if mini by 80%
        if self.mini_eval:
            data_set = data_set[::5]
        else:
            data_set = data_set

        correct_answer = 0
        incorrect_entries = []
        case_accuracies = []
        for test in self.TEST_CASES:
            correct_case_answer = 0
            print(f"Testing: {test['human_prompt']}")
            labels = test['evaluation_function'](data_set)
            for i, data in enumerate(tqdm.tqdm(data_set)):
                prompt = self.build_prompt(human_prompt=test['human_prompt'], robot_state=data)

                # Get the model's response
                if self.local_inference:
                    llm_response, _, _ = self.llm(prompt)
                else:
                    llm_response = self.llm.invoke(prompt).content
                llm_output = self.sanitize_output(llm_response)

                # Evaluate the model's response
                if llm_output == labels[i]:
                    correct_answer += 1
                    correct_case_answer += 1
                else:
                    # Log incorrect test case
                    incorrect_entries.append({
                        "test_case": test['human_prompt'],
                        "sample_index": i,
                        "prompt": prompt,
                        "model_response": llm_response,
                        "sanitized_output": llm_output,
                        "expected_output": labels[i]
                    })
            case_accuracy = correct_case_answer / len(data_set)
            case_accuracies.append(case_accuracy)
            print(f"Case {test['human_prompt']} accuracy: {case_accuracy:.2%}")

        accuracy = correct_answer / (len(data_set) * len(self.TEST_CASES))
        print(f"Total Accuracy for data: {data_dir}: {accuracy:.2%}")

        # Log relevant information
        with open(log_file, 'w') as f:
            f.write("### Case Accuracies ###\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Using RAG: {self.use_rag}\n")
            f.write(f"Using Qantization: {self.quant}\n")
            for i, case in enumerate(self.TEST_CASES):
                f.write(f"Case {case['human_prompt']} accuracy: {case_accuracies[i]:.2%}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total Accuracy for data: {data_dir}: {accuracy:.2%}\n")
            f.write("-" * 50 + "\n")
            f.write("### Incorrect Test Cases Log ###\n")
            for entry in incorrect_entries:
                f.write(f"Test Case: {entry['test_case']}\n")
                f.write(f"Sample Index: {entry['sample_index']}\n")
                f.write(f"Prompt: {entry['prompt']}\n")
                f.write(f"Model Response: {entry['model_response']}\n")
                f.write(f"Sanitized Output: {entry['sanitized_output']}\n")
                f.write(f"Expected Output: {entry['expected_output']}\n")
                f.write("-" * 50 + "\n")
        print(f"Logged to {log_file}")

if __name__ == '__main__':
    # Fetch all valid dataset names by removing the `.json` extension
    possible_datasets = sorted([
        os.path.splitext(p=file)[0] 
        for file in os.listdir(path="tests/decision_tester/robot_states") 
        if file.endswith('.json')
    ])
    
    # Fetch local models from the models directory
    local_models = [os.path.join('models', f) for f in os.listdir(path='models')]
    
    parser = argparse.ArgumentParser(description='Test the reasoning pipeline on a single scenario.')
    parser.add_argument('--model', type=str, default='gpt-4o', choices=['gpt-4o', 'unsloth/Qwen2.5-7B-Instruct', 'unsloth/Phi-3-mini-4k-instruct', 'nibauman/RobotxLLM_Qwen7B_SFT'] + local_models, help='Choose the model to use.')
    parser.add_argument('--rag', action='store_true', help='Whether to use RAG.')
    parser.add_argument(
        '--dataset',
        type=str,
        default='all',
        choices=['all'] + possible_datasets,
        help=f"Choose the dataset to use. Options are: all, {', '.join(possible_datasets)}"
    )
    parser.add_argument('--mini', action='store_true', help='Whether to run a mini test.')
    parser.add_argument('--quant', action='store_true', help='If you want to use Q5')
    args = parser.parse_args()

    load_dotenv(dotenv_path=find_dotenv())
    OPENAI_API_TOKEN = os.getenv(key="OPENAI_API_TOKEN")

    # define model
    llm = None
    local = False
    if args.model == 'gpt-4o':
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model_name='gpt-4o', openai_api_key=OPENAI_API_TOKEN)
    else:
        local = True
        # if args.model is not locally available, then pull from HuggingFace
        model_dir = args.model
        chat_template = 'qwen-2.5' if 'qwen' in args.model else 'phi-3'
        if args.quant:
            from inference.inf_gguf import RaceLLMGGGUF
            # Find gguf in model_dir
            gguf_name = [f for f in os.listdir(model_dir) if f.endswith('.gguf')][0]
            llm = RaceLLMGGGUF(model_dir=model_dir, gguf_name=gguf_name)
            print(f"Using model {gguf_name} from {model_dir}")
        else:
            from inference.inf_pipeline import RaceLLMPipeline
            llm = RaceLLMPipeline(model_dir=model_dir, load_in_4bit=True, chat_template=chat_template)
            print(f"Using model {args.model} from {model_dir}")

    # Evaluate the decision making on all datasets
    if args.dataset == 'all':
        evaluator = DecisionTester(llm=llm, model_name=args.model, all_tests=True, mini=args.mini, local=local, use_rag=args.rag, quant=args.quant)
        for i, dataset in enumerate(possible_datasets):
            data_dir = os.path.join('tests/decision_tester/robot_states', dataset + '.json')
            # Evaluate the decision making
            evaluator.eval_decision_making(data_dir=data_dir, llm=llm, data_name=dataset)
    # Only evaluate on a specific dataset
    else:
        evaluator = DecisionTester(llm=llm, model_name=args.model, all_tests=False, mini=args.mini, local=local, use_rag=args.rag, quant=args.quant)
        data_dir = os.path.join('tests/decision_tester/robot_states', args.dataset + '.json')
        # Evaluate the decision making
        evaluator.eval_decision_making(data_dir=data_dir, llm=llm, data_name=args.dataset)
