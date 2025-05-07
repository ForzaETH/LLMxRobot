## 📚 **Overview**

`DecisionTester` is an evaluation framework designed to assess the reasoning capabilities of an AI model embedded in an autonomous racing car. It evaluates the car's behavior against various test scenarios using predefined evaluation functions and logs the results for further analysis.

---

## 🚀 **Usage**

### **Run Evaluation on All Datasets**

```bash
python3 -m tests.decision_tester.decision_tester --model nibauman/RobotxLLM_Qwen7B_SFT --dataset all --mini --rag
```

### **Available Arguments:**
- `--model`: Choose the model (`gpt-4o` or `nibauman/RobotxLLM_Qwen7B_SFT` or any model available from [unsloth](https://huggingface.co/unsloth)).  
- `--dataset`: Choose the dataset (`all`, `stop`, `reverse`, etc.).  
- `--mini`: Run a reduced dataset (`--mini` enables mini evaluation mode).  
- `--quant`: Use GGUF quantized model (`Q5`) for faster inference.
- `--rag`: Uses the RAG for hints.

---

## 📊 **Available Test Scenarios**

| Test Case              | Description                       |
|-------------------------|-----------------------------------|
| **Don't move**         | Ensures the car remains stationary. |
| **Reverse the car**    | Checks if the car moves backward.  |
| **Drive forward**      | Validates forward movement.       |
| **Oscillate!**         | Detects irregular lateral motion. |
| **Drive close to the left wall** | Measures wall proximity. |
| **Drive on the centerline** | Tests alignment with the centerline. |
| **Drive faster than 3 m/s** | Verifies speed threshold. |
| **Drive on the racing line** | Ensures adherence to the optimal path. |

---

## 📁 **Log Files**

- Logs are saved in the `logs` directory with timestamped filenames:
  ```
  tests/decision_tester/logs/{model}_{dataset}_{timestamp}.txt
  ```
- Includes:
  - **Case Accuracies:** Success rates per scenario.  
  - **Incorrect Entries:** Detailed logs of mismatches, including prompts and responses.

---

## 🛡️ **Environment Variables**

Ensure the following are set in your `.env` file if you want to use GPT4o but also needed for RAG embeddings:
```
OPENAI_API_TOKEN=your_openai_key
```

---

Happy Testing! 🚗💨