<h1 align="center">
 
  SEAD: Self-Evolving Agent for Service Dialogue
</h1>

<div align="center">

<p><em>A co-evolutionary reinforcement learning framework for training dialogue agents that adapt to diverse user scenarios without requiring additional training data.</em></p>

[![Huggingface](https://img.shields.io/badge/Huggingface-Model-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/dayll/SEAD-14B)
[![Github](https://img.shields.io/badge/Github-SEAD-000000?style=for-the-badge&logo=github)](https://github.com/Da1yuqin/SEAD)
[![Paper](https://img.shields.io/badge/Paper-2602.03548-A42C25?style=for-the-badge)](https://arxiv.org/abs/2602.03548)
</div>

![mainfig](assets/framework.png)

**SEAD (Self-Evolving Agent for Service Dialogue)** - Co-evolutionary Training Loop Framework. The controller samples initial states (Phase 1), which initialize dialogues producing trajectories (Phase 2), used to train the agent with rewards (Phase 3) and compute completion rates (Phase 4), which feed back to adjust sampling distributions, closing the co-evolutionary loop.


✨If you like this project, please give it a star🌟—it's the best encouragement for us🥺!✨

---

## 🚀 More from Our Team

SEAD is part of our broader effort to build service agents that can **understand users, act strategically, and succeed under real-world constraints**—from rigorous evaluation to cost-aware reinforcement learning and proactive interaction.

| Work | Venue | What it does |
|---|---|---|
| **SAGE: A Service Agent Graph-guided Evaluation Benchmark**<br>[![arXiv](https://img.shields.io/badge/arXiv-2604.09285-b31b1b.svg)](https://arxiv.org/abs/2604.09285) | **EMNLP 2026 Findings** | Puts service agents to the test with dynamic dialogue graphs that measure both logical compliance and interaction quality. |
| **Unlocking Proactivity in Task-Oriented Dialogue**<br>[![arXiv](https://img.shields.io/badge/arXiv-2605.22240-b31b1b.svg)](https://arxiv.org/abs/2605.22240) | **arXiv 2026** | Teaches agents to take the initiative—uncovering users' hidden concerns and proactively steering conversations toward success. |
| **Reinforcing Real-world Service Agents: Balancing Utility and Cost in Task-Oriented Dialogue**<br>[![arXiv](https://img.shields.io/badge/arXiv-2602.22697-b31b1b.svg)](https://arxiv.org/abs/2602.22697) | **ICML 2026** | Trains service agents to deliver helpful, empathetic interactions while staying within real-world operational budgets. |

---

## 🔥 News
**[2026-05-13]** — New 2-instance User Simulator training mode is now available! This mode runs 2 independent user simulator instances (1 GPU each) instead of 1 instance (2 GPUs), significantly reducing the risk of user simulator crashes and improving training stability. 🚀

**[2026-04-07]** — SEAD has been accepted to ACL 2026 Findings!

**[2026-02-04]** — We've open-sourced our full research stack!  
- 📄 **Paper (arXiv)**: [arXiv:2602.03548](https://arxiv.org/abs/2602.03548)
- 💻 **Code (GitHub)**: Complete training, inference, and evaluation pipelines
- 🤗 **Model (Hugging Face)**: [dayll/SEAD-14B](https://huggingface.co/dayll/SEAD-14B)
- 📊 **Benchmark**: Benchmark and evaluation code is available

**What's included:**
- ✅ End-to-end training, inference, and evaluation pipelines
- ✅ Reproducible configs and scripts
- ✅ Pretrained checkpoints (14B parameters)
- ✅ Comprehensive evaluation suite
- ✅ Clear documentation and examples

---

## 📋 Table of Contents

- [✨ Highlights](#-highlights)
- [⚙️ Features](#️-features)
- [🤖 Model](#-model)
- [🏆 Performance](#-performance)
- [⬇️ Installation](#️-installation)
- [🚀 Quick Start](#-quick-start)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [🙏 Acknowledgements](#-acknowledgements)
- [📖 Citation](#-citation)

---

## ✨ Highlights

🎯 **Zero Training Data Required**: Our co-evolutionary framework eliminates the need for manually collected dialogue data

🚀 **State-of-the-Art Performance**: Achieves 52.0% completion rate, outperforming GPT-4o (44.2%) with only 14B parameters

💰 **Cost-Effective**: Zero inference cost compared to commercial APIs (GPT-4o: ¥727.28 for 1000 samples)

🔄 **Self-Evolving**: Automatic curriculum learning through adaptive state sampling

⚡ **Efficient Training**: Supports distributed training on 8 GPUs with vLLM acceleration

---

## ⚙️ Features

### 🎓 Training & Optimization
- ✅ **Co-evolutionary Framework**: Adaptive curriculum learning via state controller
- ✅ **Distributed Training**: Multi-GPU support with efficient parallelization
- ✅ **Checkpoint Management**: Automatic saving and resuming
- ✅ **User Response Cache**: Accelerates training by caching User Simulator responses, reducing redundant vLLM calls

### 💾 User Response Cache

To accelerate training, we introduce a `UserResponseCache` (implemented in `SEAD/llm_agent/generation.py`) that caches User Simulator responses. Instead of calling the vLLM-served user model for every single dialogue turn, the cache reuses previously generated responses at configurable hit rates.

**How it works:**
- Responses are cached per dialogue stage (0–3), each with a **stage-specific hit rate**:
  - **Stage 0 (Opening phase)**: 90% — highly homogeneous, safe to reuse
  - **Stage 1 (Objection handling)**: 50% — diverse user behaviors, moderate reuse
  - **Stage 2 (Cooperation phase)**: 20% — needs personalization, low reuse
  - **Stage 3 (Confirmation step)**: 0% — always calls vLLM for fresh responses
- To prevent stale conversations, the cache enforces a rule: **no two consecutive turns** for the same sample can both use cached responses (at least one real vLLM call in between).
- Cache is persisted to disk in JSONL format and auto-loaded on restart, with automatic cleanup when exceeding size limits.

**To disable the cache** (e.g., for debugging or if you want fully fresh responses every turn), set all stage hit rates to `0` in `generation.py`:

```python
self.stage_hit_rates = {
    '0': 0.0,
    '1': 0.0,
    '2': 0.0,
    '3': 0.0,
}
```



## 🤖 Model
SEAD is now available on huggingface-hub:
| Model Name | HF Checkpoint                                                | Size                                                    |
| ---------- | ------------------------------------------------------------ | :------: |
| SEAD-14b     | [🤗 dayll/SEAD-14B](https://huggingface.co/dayll/SEAD-14B) | **14B** 



## 🏆 Performance
## Experimental Results

### Main Results Comparison

| Method | Params | CR (%) | ATT ↓ | UPA | EI | TI | CI | Total Cost (CNY) |
|--------|--------|--------|-------|-----|----|----|----|--------------------|
| **Foundation Models** |
| Qwen2.5-14B-Instruct | 14B | 38.7 | 10.5±2.1 | 0.883±0.085 | 0.34±1.11 | 0.68±1.53 | 0.63±1.58 | 0.00 |
| Qwen2.5-32B-Instruct | 32B | 38.3 | 9.9±2.15 | 0.899±0.068 | -0.11±0.54 | 0.76±0.91 | 2.25±1.15 | 0.00 |
| Qwen2.5-72B-Instruct | 72B | 39.0 | **9.6±2.18** | 0.818±0.144 | <ins>0.51±1.32</ins> | 1.06±1.72 | 1.18±1.59 | 0.00 |
| **Large Model APIs** |
| GPT-4o | -- | <ins>44.2</ins> | 10.8±2.10 | 0.867±0.117 | 0.04±0.97 | 0.97±1.29 | 1.34±1.42 | 727.28 |
| DeepSeek-Chat | 671B | 31.6 | 11.3±2.10 | 0.863±0.084 | -0.20±0.97 | 0.27±1.24 | 0.76±1.50 | 87.36 |
| Qwen3-235B | 235B | 32.3 | 10.4±2.50 | 0.765±0.170 | -0.24±0.83 | 0.80±1.14 | 1.54±1.50 | 69.36 |
| LongCat-Flash | 560B | 42.2 | 10.0±2.31 | **0.925±0.079** | 0.28±1.15 | <ins>1.33±1.57</ins> | **1.56±1.46** | 23.08 |
| **SEAD (Ours)** | **14B** | **52.0** | **9.6±2.09** | <ins>0.912±0.071</ins> | **0.63±1.12** | **1.57±1.51** | <ins>1.55±1.39</ins> | **0.00** |

**Metrics:**
- **Params**: Model parameters (B=billion, "--" indicates undisclosed or not applicable)
- **CR**: Completion Rate (%)
- **ATT**: Average Turns to Target (lower is better ↓)
- **UPA**: User Portrait Accuracy
- **EI**: Emotion Improvement
- **TI**: Trust Improvement
- **CI**: Cooperation Improvement
- **Total Cost**: Total inference cost for 1000 multi-turn samples (CNY)

**Note**: **Bold** indicates best results. <ins>Underlined</ins> indicates second-best results. Standard deviations are shown where available.

### Dynamic Training Results
![mainfig](assets/chatbot_metrics.png)
With the advancement of training, the model's metrics steadily improve, highlighting the effectiveness of RL. The hard business metric, Task Completion, achieves a significant boost, showing that the model has learned better strategies through free exploration. The increase in User Profile Accuracy demonstrates that the model understands users better, while the steady rise in the Trust Variation Mean indicates that the model can more easily gain user trust through conversation.

## ⬇️ Installation
### Environment
```bash
conda create -n SEAD python=3.10
conda activate SEAD
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
pip install wandb
```

Alternatively, you can configure the environment according to `requirements.txt`:
```bash
pip install -r requirements.txt
```


The User Role-play Model operates within an isolated environment.

```bash
conda create -n vllm python=3.10
pip install -r requirements_vllm.txt
```


## 🚀 Quick start
To modify prompts (such as user profiles and SOP), edit the files in: ``./verl/trainer/config/format_prompt/``. 

Common user behaviors can be modified by editing
``./assets/client_action.jsonl``. These behaviors are randomly sampled and incorporated into user prompts to ensure dialogue diversity."
### Training
Our model requires no additional training data. Simply load the base model to start training.

We provide two training modes. **2-instance mode is recommended** for better stability.

#### Mode 1: 2-instance User Simulator (Recommended ✅)

Runs 2 independent user simulator instances, each on 1 GPU (GPU 6, 7), listening on ports **5000** and **5001** respectively. This avoids single-instance overload and greatly reduces crash risk.

**How it works**: In `generation.py`, the chatbot URL is configured as a comma-separated list (`http://localhost:5000,http://localhost:5001`). During each training step, the batch of dialogue requests is evenly split and dispatched to both instances in parallel via multi-threading, then results are merged back. Since each instance runs independently on a single GPU, this eliminates **NCCL communication timeouts** that can occur when a single instance spans multiple GPUs under heavy load.

```bash
conda activate SEAD
bash ./scripts/2ins_main.sh
```

GPU allocation: Chatbot training (GPU 0-5, 6 cards) + User Sim 1 (GPU 6) + User Sim 2 (GPU 7)

#### Mode 2: 1-instance User Simulator

Runs 1 user simulator instance on 2 GPUs (GPU 6, 7) with a single chatbot (1 user + 1 chatbot). Simpler setup with fewer moving parts — if you want to modify or extend the codebase, this mode is easier to start with. However, the single instance may crash under heavy load due to NCCL communication timeouts.

```bash
conda activate SEAD
bash ./scripts/main.sh
```

GPU allocation: Chatbot training (GPU 0-5, 6 cards) + User Sim (GPU 6-7, 2 cards)

#### Training Configuration

Edit `./scripts/main.sh` or `./scripts/2ins_main.sh` to customize:
- Base model path

Edit `./scripts/train_chatbot.sh` or `./scripts/2ins_train_chatbot.sh` to customize:
- Batch size and learning rate
- Checkpoint save frequency

#### Stop Training

To stop a running training process:

```bash
pkill -9 -f "main.sh"
pkill -9 -f "train_chatbot.sh"
pkill -9 -f "generate_profiles_parallel.py"
ray stop --force
ps aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
```

To visualize dynamic curves locally, run the following command:

```bash
python for_evaluation/metrics_vis.py
```

The generated plots will be saved in `./outputs/evaluation/report`.

### Evaluation
Test any local model or your custom-trained model:

```bash
# Create Evaluation Set
python utils/create_prompt_data.py \
    --train_samples 0 \
    --test_samples 1000 \
    --behavior_library ./assets/client_action.jsonl \
    --out_dir ./outputs/evaluation/test_set/ \
    --temp_dir ./outputs/evaluation/test_set/user_param/
# Run the evaluation following the instructions in the log
bash ./for_evaluation/vllm_test_suite.sh
``` 
Modify ./for_evaluation/vllm_test_suite.sh to set:
- Model checkpoint path



## 🙏 Acknowledge

The SEAD framework draws inspiration from pioneering projects such as [Search-R1](https://github.com/PeterGriffinJin/Search-R1), is built upon [veRL](https://github.com/volcengine/verl) and [RAGEN](https://github.com/ZihanWang314/RAGEN/tree/main). 
We would like to sincerely thank the teams behind these projects for their invaluable contributions to open-source research and development.

## 🏷️ Citation
```bash
@article{SEADv1,
  title={SEAD: Self-Evolving Agent for Multi-Turn Service Dialogue},
  author={Yuqin Dai, Ning Gao, Wei Zhang, Jie Wang, Zichen Luo, Jinpeng Wang, Yujie Wang, Ruiyuan Wu, Chaozheng Wang},
  journal={arXiv preprint arXiv:2602.03548},
  year={2026}
}
```
