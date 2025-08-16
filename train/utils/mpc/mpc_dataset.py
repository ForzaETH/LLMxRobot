import os
from torch.utils.data import Dataset
import random
from typing import List, Dict, Any, Callable, Union
from langchain_core.documents import Document

class MPCDatasetGRPO(Dataset):
    """
    Dataset class for training an LLM with GRPO on autonomous racing car decisions.
    Generates prompts from robot state data and human prompts without answers,
    as the correct answers will be determined by the reward function during training.
    """
    def __init__(self, 
                 test_cases: List[Dict[str, Union[str, Callable]]],
                 use_rag: bool = False,
                 index = None,
                 mem_nb: int = 5,
                 shuffle: bool = True):
        """
        Initialize the dataset with the prompts and test cases.
        
        Args:
           
        """
        # Load base MPC memory
        # Base Memory
        base_mem_dir = os.path.join('./', 'prompts/mpc_base_memory.txt')
        print(f'Loading base memory from {base_mem_dir}...')
        with open(base_mem_dir, 'r') as f:
            base_mem = f.read()
        self.base_memory = base_mem
        
        # save params
        self.memory_nb = mem_nb
        self.use_rag = use_rag
        self.index = index
        self.test_cases = test_cases
            
        # Create combinations of robot states and human prompts
        self.samples = []
        for test_case in self.test_cases:
            self.samples.append({
                "human_prompt": test_case["human_prompt"],
                "evaluation_function": test_case["evaluation_function"],
                "baseline_rmse": test_case["baseline_rmse"],
                "ref_val": test_case.get("ref_val", None),
            })
        
        # Shuffle if requested
        if shuffle:
            random.Random(42).shuffle(self.samples)
            
        print(f"MPC dataset created with {len(self.samples)} samples")
            
    def build_prompt(self, human_prompt: str) -> str:
        """
        Build a prompt using the template with robot state information.
        
        Args:
            human_prompt: The task requested by the human
            robot_state: Dictionary containing the robot's state information
            
        Returns:
            Formatted prompt string
        """
        # Hints are empty if not using RAG
                # Query the index and pass the result to the command chain for processing
        RAG_query = f"""
        Task: {human_prompt}\n
        """
        #Perform RAG manually
        #Retrieve docs from the RAG
        rag_sources: List[Document] = self.index.vectorstore.search(query=RAG_query, search_type='similarity', k=self.memory_nb) if self.memory_nb > 0 else []
        rag_sources = [{'meta': doc.metadata, 'content': doc.page_content} for doc in rag_sources]
        prompt = f"""
        You are an AI assistant helping to tune the parameters of an MPC controller for an autonomous racing car. Below is the context and the task:

        ## Context
        1. **Scenario**: {human_prompt}
        2. **Base Memory and Cost Formulation**: {self.base_memory}

        ## Task
        Adapt the tuneable parameters of the MPC so that the car achieves the following: **{human_prompt}**.

        ## Caution
        - Do not invent new parameters! Strictly adhere to the parameters provided in the base memory.

        ## Constraints
        - Use the min and max ranges of the parameters provided in the base memory.
        - Only consider relevant RAG information and quote it briefly in your explanation.

        ## RAG Information
        - !!! Not all memories are relevant to the task. Select the most relevant ones. !!!
        - **Memory Entries**:
        {rag_sources}

        ## Expected Output Format
        Always strictly return your answers in the following format:
        <reasoning> \n
        ...
        </reasoning> \n
        <answer> \n
        new_mpc_params = {{
            'param1': new_value1,
            'param2': new_value2,
            ...
        }}
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
        prompt = self.build_prompt(sample["human_prompt"])
        
        # Return the prompt and metadata that might be useful for evaluation
        return {
            "prompt": prompt,
            "evaluation_function": sample["evaluation_function"],
            "baseline_rmse": sample["baseline_rmse"],
            "ref_val": sample.get("ref_val", None),
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