import json
from tqdm import tqdm
import os
from torch.utils.data import Dataset
import numpy as np
import random
from typing import List, Dict, Any, Callable, Optional, Union
import train.utils.decision.eval_states as eval_states

class DecisionDatasetGRPO(Dataset):
    """
    Dataset class for training an LLM with GRPO on autonomous racing car decisions.
    Generates prompts from robot state data and human prompts without answers,
    as the correct answers will be determined by the reward function during training.
    """
    def __init__(self, 
                 raw_robot_states_dir: str,
                 test_cases: List[Dict[str, Union[str, Callable]]],
                 raw_nb_gen: int = 100,
                 use_rag: bool = False,
                 from_raw: bool = False,
                 index = None,
                 mem_nb: int = 5,
                 shuffle: bool = True,
                 max_samples: Optional[int] = None,
                 mode:str = "train"):
        """
        Initialize the dataset with robot states and test cases.
        
        Args:
            raw_robot_states_dir: Path to JSON files containing raw robot state data
            test_cases: List of dictionaries with human prompts and evaluation functions
            use_rag: Whether to use RAG for generating hints
            index: Vector store index for RAG (if use_rag is True)
            mem_nb: Number of memory items to retrieve from RAG
            shuffle: Whether to shuffle the dataset
            max_samples: Maximum number of samples to include in the dataset (for debugging or limiting dataset size)
        """
        self._sliced_path = os.path.join(raw_robot_states_dir, "sliced_states")
        if from_raw:
            self.raw_robot_states = self._slice_raw_states(raw_robot_states_dir, raw_nb_gen)
        if mode == "train":
            data_path = self._sliced_path
        else:
            data_path = raw_robot_states_dir
        self.robot_states = self._load_robot_states(data_path)
        self.test_cases = test_cases
        self.use_rag = use_rag
        self.index = index
        self.mem_nb = mem_nb
        
        # Create all possible combinations of robot states and human prompts
        self.samples = []
        for robot_state in self.robot_states:
            for test_case in self.test_cases:
                self.samples.append({
                    "robot_state": robot_state,
                    "human_prompt": test_case["human_prompt"],
                    "evaluation_function": test_case["evaluation_function"]
                })
        
        # Shuffle if requested
        if shuffle:
            random.Random(42).shuffle(self.samples)
        
        # Limit dataset size if requested
        if max_samples is not None and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]
            
        print(f"Dataset created with {len(self.samples)} samples")
    
    def _load_robot_states(self, json_dir: str) -> List[Dict[str, Any]]:
        """Find all json files in json_dir and load them into a list of dictionaries."""
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        print("++++++++++++++++++++++++++++++++++++")
        print(json_files)
        print("++++++++++++++++++++++++++++++++++++")
        robot_states = []
        for json_file in json_files:
            with open(os.path.join(json_dir, json_file), 'r') as f:
                robot_states.extend(json.load(f))
        return robot_states

    def _slice_raw_states(self, json_dir: str, raw_nb: int = 100) -> None:
        """Slice the robot states into chunks of size slice_size."""
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        print("++++++++++++++++++++++++++++++++++++")
        print(json_files)
        print("++++++++++++++++++++++++++++++++++++")
        
        for json_file in json_files:
            print(f"Processing file: {json_file}")
            with open(os.path.join(json_dir, json_file), 'r') as f:
                robot_state = json.load(f)
                
            slices = []
            for _ in tqdm(range(raw_nb)):
                # Slice the robot state into chunks of size slice_size
                slice_size = np.random.randint(3, 12)
                slice_idx = np.random.randint(0, len(robot_state["s_pos"]) - slice_size)
                sliced_state = {
                    key: value[slice_idx:slice_idx+slice_size] 
                    for key, value in robot_state.items()
                }
                sliced_state["data_samples"] = slice_size
                sliced_state["time"] = slice_size / 10  # Assuming 10 Hz data
                
                # reversing and crashed bool
                sliced_state["reversing"] = True if np.mean(sliced_state["s_speed"]) < -0.1 else False
                sliced_state["crashed"] = True if np.mean(sliced_state["d_left"]) < 0.1 or np.mean(sliced_state["d_right"]) < 0.1 else False
                slices.append(sliced_state)
            
            # Save the sliced states to a new file
            with open(os.path.join(self._sliced_path, "sliced_" + json_file), 'w') as f:
                json.dump(slices, f, indent=4)
            print(f"Saved sliced states to {self._sliced_path}")
            print("++++++++++++++++++++++++++++++++++++")
            
                
    def build_prompt(self, human_prompt: str, robot_state: Dict[str, Any]) -> str:
        """
        Build a prompt using the template with robot state information.
        
        Args:
            human_prompt: The task requested by the human
            robot_state: Dictionary containing the robot's state information
            
        Returns:
            Formatted prompt string
        """
        # Hints are empty if not using RAG
        hints = ''
        if self.use_rag:
            rag_sources = self.index.vectorstore.search(query=human_prompt, search_type='similarity', k=self.mem_nb) if self.mem_nb > 0 else []
            rag_sources = [{'meta': doc.metadata, 'content': doc.page_content} for doc in rag_sources]
            for hint in rag_sources:
                hints += hint['content'] + "\n"
        # print("++++++++++++++++++++++++++++++++++++++++++")
        # print(robot_state)
        # print("++++++++++++++++++++++++++++++++++++++++++")

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
        Use these guides to reason: \n\n{hints}\n\n    
        Check if the car is adhering to what the human wants: {human_prompt}. Strictly reply in the following format: \n
        <reasoning> \n
        ...
        </reasoning> \n
        <explanation> \n
        Explanation: <Brief Explanation> \n
        </explanation>
        <answer> \n
        Adhering to Human: <True/False> \n
        </answer>
        """
        return prompt
    
    def __len__(self) -> int:
        """Return the number of train samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the training split of the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing the prompt and metadata
        """
        sample = self.samples[idx]
        prompt = self.build_prompt(sample["human_prompt"], sample["robot_state"])
        
        # Return the prompt and metadata that might be useful for evaluation
        return {
            "prompt": prompt,
            "robot_state": sample["robot_state"],
            "evaluation_function": sample["evaluation_function"]
        }
    
    def get_batch(self, batch_size: int, start_idx: int = 0) -> List[Dict[str, Any]]:
        """
        Get a batch of train samples from the dataset.
        
        Args:
            batch_size: Number of samples to include in the batch
            start_idx: Starting index for the batch
            
        Returns:
            List of dictionaries containing prompts and metadata
        """
        end_idx = min(start_idx + batch_size, len(self.samples))
        batch = []
        
        for idx in range(start_idx, end_idx):
            batch.append(self.__getitem__(idx))
            
        return batch
                        
    def save_to_json(self, output_file: str) -> None:
        """
        Save the dataset to a JSON file for later use. The JSON file will contain a list of dictionaries,
        each with a prompt and metadata."""
        with open(output_file, 'w') as f:
            json.dump([self.__getitem__(idx) for idx in range(len(self))], f, indent=2)

if __name__ == "__main__":
    # Example usage
    raw_robot_states_dir = "/embodiedai/train/dataset/grpo_decision/raw_states"

    # Create dataset
    dataset = DecisionDatasetGRPO(
        raw_robot_states_dir=raw_robot_states_dir,
        raw_nb_gen=250,
        test_cases=eval_states.EVAL_CASES,
        use_rag=False,  # Set to True if using RAG
        shuffle=True
    )

    # Get a single sample
    sample = dataset[0]
    print(sample["prompt"])

    # Get a batch of samples
    batch = dataset.get_batch(batch_size=4)

    # Save dataset to JSONL for later use
    dataset.save_to_json("/embodiedai/train/dataset/grpo_decision/decision_dataset_from_drive.json")