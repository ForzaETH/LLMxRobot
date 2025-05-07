from llama_cpp import Llama
import os

class RaceLLMGGGUF:
    def __init__(self, model_dir, gguf_name, max_tokens=256):
        self.max_tokens = max_tokens
        self.path = os.path.join(model_dir, gguf_name)
        self.llm = Llama(
            model_path=self.path,
            chat_format="llama-2",
            n_gpu_layers=1000,
            n_ctx=2048,
            n_batch=2048,
            seed=42,
            verbose=False
            )

    def __call__(self, input_text):
        output = self.llm.create_chat_completion(
            messages=[{
                     "role": "user",
                     "content": input_text
                 }],
            max_tokens=self.max_tokens,
            temperature=0.0,
            top_k=1,
            stop=["[/INST]", "[\/INST]", "[;/INST] ", "[INST]", "[/?]", "[/Dk]", "[;/Rationale]", "[Rationale]", "[;/Action]", "[;/Explanation]"],
        )
        out_text = output['choices'][0]['message']['content']
        input_tokens = output['usage']['prompt_tokens']
        out_tokens = output['usage']['completion_tokens']
        return out_text, input_tokens, out_tokens

# Loads Prompt with hints
def load_prompt(prompt_type) -> str:
    if 'reasoning' in prompt_type:
        hints_dir = os.path.join('../', 'prompts/reasoning_hints.txt')
        with open(hints_dir, 'r') as f:
            reasoning_hints = f.read()
        return reasoning_hints
    elif 'synthetic' in prompt_type:
        hints_dir = os.path.join('../', 'prompts/example_synthetic.txt')
        with open(hints_dir, 'r') as f:
            synthetic_hints = f.read()
        return synthetic_hints
    else:
        raise ValueError(f"Prompt type {prompt_type} not recognized. Please use 'reasoning' or 'synthetic'.")