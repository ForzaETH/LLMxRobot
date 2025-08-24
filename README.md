# LLMxRobot

This repository is a two in one codebase for two papers on using Large Language Models (LLMs) for autonomous driving on a small-scale robotic platform.


In our [RSS 2025 paper](https://arxiv.org/abs/2504.11514), **Enhancing Autonomous Driving Systems with On-Board Deployed Large Language Models**. We demonstrate how LLMs can be used for autonomous driving. It provides the codebase for the **MPCxLLM** and **DecisionxLLM** modules, alongside tools for training, testing, and deployment.

🚗 Small LLMs can adapt driving behavior through MPC and perform decision making:

<p align="center"\>
  <img src=".misc/rss_abstract_figure.png" width="600" alt="RobotxLLM Overview"\>
</p\>

Watch an explanatory Youtube video accompanying the paper [here](https://www.youtube.com/watch?v=4iGN1uBl4v4).

In our [CoRL 2025 paper](https://arxiv.org/abs/2505.03238), **RobotxR1: Enabling Embodied Robotic Intelligence on Large Language Models through Closed-Loop Reinforcement Learning**. We build uppon the previous robotic system and introduce closed-loop RL training of LLMs to to foster a *learning by doing* paradigm for embodied robotic intelligence. Showing that small scale LLMs can learn to drive a robot car through RL from its own experience.

📈 +20.2%-points improvement over SFT baseline with Qwen2.5-1.5B via RL.

🧠 63.3% control adaptability with Qwen2.5-3B, surpassing GPT-4o (58.5%) in this robotic task:

<p align="center"\>
  <img src=".misc/corl_abstract_figure.png" width="600" alt="RobotxLLM Overview"\>
</p\>

Check out the Wandb report of the training runs [here](https://wandb.ai/CoRL-heist-2025/mpc_grpo/reports/RobotxR1-Training-Runs--VmlldzoxNDEwOTczNQ?accessToken=a2qws19tg5fuwv7bi2mto42lbjj9l83hyz3feib3vj6x6dp25ul8j6wqhvkiuhcq).


## 🚀 Installation

### CUDA Platform (e.g., RTX 30xx / 40xx)

1. Build the Docker container (adapt `CUDA_ARCH` accordingly: `86` for RTX 30xx, `89` for 40xx):
   ```bash
   docker build --build-arg CUDA_ARCH=<your_compute_capability> -t embodiedai -f .docker_utils/Dockerfile.cuda .
   ```

2. Mount the container to the project directory:
   ```bash
   ./.docker_utils/main_dock.sh cuda
   ```

3. Attach to the container:
   ```bash
   docker exec -it embodiedai_dock /bin/bash
   ```
   or use VS Code Remote Containers.

---

### Jetson Platform (e.g., Orin AGX)

1. Build the ARM-compatible Docker image:
   ```bash
   docker build -t embodiedai -f .docker_utils/Dockerfile.jetson .
   ```
   **Note that on the jetson, unsloth can not be installed (as of 07.05.2025). So only inference via quantized models are possible!**

2. Mount and launch the container:
   ```bash
   ./.docker_utils/main_dock.sh jetson
   ```

3. Attach via terminal or VS Code.

### Create .env File
Create a `.env` file in the root directory with the following content:
```bash
HUGGINGFACEHUB_API_TOKEN="<your_huggingface_token>"
OPENAI_API_TOKEN="<your_openai_token>"
WANDB_API_KEY="<your_wandb_api_key>" # Optional
```
This is needed for downloading models and using OpenAI APIs which is required if you want to use `gpt-4o` or for using the modules with their RAG embeddings. **Make sure to keep this file private!**

### Download Pre-Trained Models (optional)
**Enhancing Autonomous Driving Systems with On-Board Deployed Large Language Models**:

You can use the LoRA + RAG SFT trained FP16 model [nibauman/RobotxLLM_Qwen7B_SFT](https://huggingface.co/nibauman/RobotxLLM_Qwen7B_SFT) directly from HuggingFace 
without having to download it locally. If you want to use the quantized model, you can download it with the following command:

```bash
huggingface-cli download nibauman/race_llm-Q5_K_M-GGUF --local-dir models/race_llm_q5
```
**RobotxR1: Enabling Embodied Robotic Intelligence on Large Language Models through Closed-Loop Reinforcement Learning**:

The SFT and GRPO trained models for DecisionxLLM and MPCxLLM are available on HuggingFace:
- DecisionxLLM Qwen-1.5B: [nibauman/DecisionxR1_Qwen1.5B_SFT_GRPO](https://huggingface.co/nibauman/DecisionxR1_Qwen1.5B_SFT_GRPO)
- DecisionxLLM Qwen-3B: [nibauman/DecisionxR1_Qwen3B_SFT_GRPO](https://huggingface.co/nibauman/DecisionxR1_Qwen3B_SFT_GRPO)
- MPCxLLM Qwen-1.5B: [nibauman/MPCxR1_Qwen1.5B_SFT_GRPO](https://huggingface.co/nibauman/MPCxR1_Qwen1.5B_SFT_GRPO)
- MPCxLLM Qwen-3B: [nibauman/MPCxR1_Qwen3B_SFT_GRPO](https://huggingface.co/nibauman/MPCxR1_Qwen3B_SFT_GRPO)
---

## 🧠 Usage

This repo integrates with the [ForzaETH Race Stack](https://github.com/ForzaETH/race_stack). Follow their installation instructions and ensure your `ROS_MASTER_URI` is correctly configured (see [example line](https://github.com/ForzaETH/race_stack/blob/main/.devcontainer/.install_utils/bashrc_ext#L12)) in this readme we use 192.168.192.75 as an example!

### On the Robot Stack
Run each command in a separate terminal. Use `f` map for evaluation and `circle` map (`map_name:=circle`) for RobotxR1 RL training.
```bash
roscore
roslaunch stack_master base_system.launch map_name:=f racecar_version:=NUC2 sim:=true
roslaunch stack_master time_trials.launch ctrl_algo:=KMPC
roslaunch rosbridge_server rosbridge_websocket.launch address:=192.168.192.75
```

### On the LLM Machine

```bash
python3 llm_mpc.py --model custom --model_dir nibauman/RobotxLLM_Qwen7B_SFT --hostip 192.168.192.75 --prompt "Drive in Reverse!"
```

**Key Options:**

- `--model`: `custom` or `gpt-4o`
- `--model_dir`: HuggingFace or local path (used for `custom`)
- `--hostip`: ROS master IP
- `--prompt`: Natural language instruction
- `--quant`: Use quantized `GGUF` model
- `--mpconly`: Skip DecisionxLLM

As an **example** for on the **Jetson** you can only run the quantized models with the models downloaded to the models folder as explained above. You can run the following command to test the quantized model:
```bash
python3 llm_mpc.py --model custom --model_dir models/race_llm_q5 --hostip 192.168.192.75 --prompt "Drive in Reverse!" --quant
```
---

## 🏋️ Training

To train a new LoRA adapter on synthetic data with **Supervised Fine-Tuning (SFT)** akin to *Enhancing Autonomous Driving Systems with On-Board Deployed Large Language Models* [here](https://arxiv.org/abs/2504.11514), you can use the following command:

```bash
python3 -m train.sft_train --config train/config/sft_train.yaml
```

You can modify `sft_train.yaml` to change training parameters. Default setup:

- Base Model: `unsloth/Qwen2.5-7B-Instruct`

To train the DecisionxLLM through **static Reinforcement Learning** akin to *RobotxR1: Enabling Embodied Robotic Intelligence on Large Language Models through Closed-Loop Reinforcement Learning* [here](https://arxiv.org/abs/2505.03238), you can use the following command:

```bash
python3 -m train.rl_decision_train --config train/config/rl_decision_train.yaml
```

Modify `rl_decision_train.yaml` to change training parameters. Default setup:
- Base Model: `Qwen/Qwen2.5-3B-Instruct`

To train the MPCxLLM through **feedback driven Reinforcement Learning** akin to *RobotxR1: Enabling Embodied Robotic Intelligence on Large Language Models through Closed-Loop Reinforcement Learning* [here](https://arxiv.org/abs/2505.03238), you can use the following command:

```bash
python3 -m train.rl_mpc_train --config train/config/rl_mpc_train.yaml
```
Modify `rl_mpc_train.yaml` to change training parameters. Default setup:
- Base Model: `Qwen/Qwen2.5-3B-Instruct`

**Note:** Train the model on the `circle`map, then evaluate it on the `f` map. Command to launch the robot stack on the `circle` map:
```bash
roslaunch stack_master base_system.launch map_name:=circle racecar_version:=NUC2 sim:=true
```    
---

## 📊 Evaluation

### DecisionxLLM Evaluation (autonomy stack not required)

```bash
python3 -m tests.decision_tester.decision_tester --model nibauman/RobotxLLM_Qwen7B_SFT --dataset all --mini --rag
```

### MPCxLLM Evaluation (requires autonomy stack)

```bash
python3 -m tests.mpc_tester.mpc_tester --model custom --model_dir nibauman/RobotxLLM_Qwen7B_SFT --host_ip 192.168.192.75
```

**Evaluation Options:**

- `--dataset`: e.g., `all`, `stop`, `reverse`, etc.
- `--mini`: Run a small evaluation subset
- `--rag`: Enable retrieval-augmented decision prompts
- `--quant`: Use quantized model
---
## Acknowledgements
SFT training was performed through the distillation of [OpenAI GPT-4o](https://openai.com/index/hello-gpt-4o/) queries.
This work would not have been possible without the great work of other repositories such as:
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Hugging Face](https://github.com/huggingface)
- [unsloth](https://github.com/unslothai/unsloth)
- [roslibpy](https://github.com/gramaziokohler/roslibpy)
---

## 📄 Citation

If this repository is useful for your research, please consider citing our work:

**Enhancing Autonomous Driving Systems with On-Board Deployed Large Language Models:**
```bibtex
@article{baumann2025enhancing,
  title={Enhancing Autonomous Driving Systems with On-Board Deployed Large Language Models},
  author={Baumann, Nicolas and Hu, Cheng and Sivasothilingam, Paviththiren and Qin, Haotong and Xie, Lei and Magno, Michele and Benini, Luca},
  journal={arXiv preprint arXiv:2504.11514},
  year={2025}
}
```
**RobotxR1: Enabling Embodied Robotic Intelligence on Large Language Models through Closed-Loop Reinforcement Learning:**
```bibtex
@misc{boyle2025robotxr1enablingembodiedrobotic,
      title={RobotxR1: Enabling Embodied Robotic Intelligence on Large Language Models through Closed-Loop Reinforcement Learning}, 
      author={Liam Boyle and Nicolas Baumann and Paviththiren Sivasothilingam and Michele Magno and Luca Benini},
      year={2025},
      eprint={2505.03238},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.03238}, 
}
```
