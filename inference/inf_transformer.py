from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os

class RaceLLMTransformer:
    def __init__(self, model_dir, max_seq_length=2048, max_new_tokens=512, chat_template="phi-3", dtype=torch.float16, load_in_4bit=False):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoids a warning
        self.model_dir = model_dir
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.chat_template = chat_template

        # Config for Quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=dtype,
        )

        # Load model and tokenizer from transformers
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            device_map="auto",  # Automatically assigns layers to devices
        )
        self.tokenizer : AutoTokenizer = AutoTokenizer.from_pretrained(model_dir)

         # Add pad and eos tokens if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({'eos_token': '<|end|>'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Move model to CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def __call__(self, message):
        # Preprocess the input using a chat template
        inputs = self.preprocess(message)

        # Perform the forward pass
        outputs = self._forward(inputs)

        # Postprocess the output
        return self.postprocess(outputs, message)

    def preprocess(self, text):
        # Prepare messages based on the chat template
        if self.chat_template == "phi-3":
            messages = [{"from": "human", "value": text}]
        elif self.chat_template == "qwen-2.5":
            messages = [{"role": "user", "content": text}]
        else:
            raise ValueError(f"Chat template {self.chat_template} not recognized.")

        # Tokenize and construct inputs
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # If `encoded` is already a Tensor
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded.to(self.device)
        else:
            # Otherwise, handle the structure with `input_ids` and `attention_mask`
            input_ids = encoded["input_ids"].to(self.device)

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    def _forward(self, model_inputs, **forward_params):
        # Generate model outputs
        outputs = self.model.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            temperature=1.0,
            do_sample=False, # Greedy decoding ignores `temperature`, `top_k`, and `top_p`
            top_k=20,
            top_p=0.8,
            repetition_penalty=1.0,
            encoder_repetition_penalty=1.0,
            length_penalty=-1.0,
            num_beams=2,
        )
        return outputs

    def postprocess(self, outputs, input_text):
        # Decode outputs and filter response
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        if self.chat_template == "phi-3":
            model_outputs = [output.split("<|assistant|>")[1].split("<|endoftext|>")[0].strip() for output in decoded]
        elif self.chat_template == "qwen-2.5":
            model_outputs = [
                output.split("<|im_start|>assistant")[1].split("<|im_end|>")[0].strip() 
                if "<|im_start|>assistant" in output and "<|im_end|>" in output
                else output  # Fallback to raw output if markers are missing
                for output in decoded
            ]
        
        # TODO: This slows down the inference
        # Tokenize input and output text
        input_tokens = self.tokenizer.tokenize(input_text)
        output_tokens = self.tokenizer.tokenize(model_outputs[0])
        
        return model_outputs[0], len(input_tokens), len(output_tokens)
