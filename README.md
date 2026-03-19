# OpenVLA-OFT System2

> **This repository is forked from [openvla/openvla](https://github.com/openvla/openvla) and builds upon the work of [OpenVLA-OFT](https://openvla-oft.github.io/) by Moo Jin Kim, Chelsea Finn, and Percy Liang (Stanford University).**
> Original paper: [Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success](https://arxiv.org/abs/2502.19645)

---

## What is This?

This fork extends OpenVLA-OFT with a **System2 reasoning layer** — a two-stage hierarchical control architecture for robot manipulation:

- **System 1 (Fast):** The VLA (Vision-Language-Action) model generates low-level action chunks in real time.
- **System 2 (Slow):** A reasoning LLM (e.g., `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B`) decomposes the task into subgoals, guiding System 1 at a higher level.

This approach lets the robot break complex manipulation tasks (e.g., *"pick up the red block and place it on the green plate"*) into interpretable intermediate steps, improving robustness and generalization.

---

## Key Additions Over the Original

| Component | File | Description |
|-----------|------|-------------|
| `System2Agent` class | `experiments/robot/libero/system2_agent.py` | Reasoning agent for hierarchical task decomposition |
| Observation summarizer | `system2_agent.py` | Converts robot state (gripper, objects) to natural language |
| Subgoal generator | `system2_agent.py` | Few-shot prompting to produce the next step |
| LIBERO integration | `experiments/robot/libero/run_libero_eval.py` | System2 used during evaluation |
| Standalone test | `test_system2_standalone.py` | Quick validation of the System2 agent |

---

## System Requirements

**Inference:**
- 1 GPU with ~16 GB VRAM (LIBERO simulation tasks)
- 1 GPU with ~18 GB VRAM (ALOHA real-robot tasks)

**Training:**
- 1–8 GPUs with 27–80 GB VRAM (default bfloat16)
- See the [original FAQ](https://openvla-oft.github.io/#train-compute) for details.

**System2 Agent:**
- An additional model (e.g., 1.5B–7B LLM) runs on any available device (CUDA / MPS / CPU).
- Default model: `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B`

---

## Quick Start

### 1. Set Up Environment

```bash
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft
pip3 install torch torchvision torchaudio
git clone https://github.com/15935648a/openvla-oft-system2.git
cd openvla-oft-system2
pip install -e .
# Install Flash Attention 2
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

### 2. Run System2 Agent (Standalone Test)

```bash
python test_system2_standalone.py
```

This will load the reasoning model and demonstrate subgoal generation from a task description.

### 3. Run VLA Inference

```python
import pickle
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

cfg = GenerateConfig(
    pretrained_checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial",
    use_l1_regression=True,
    use_diffusion=False,
    use_film=False,
    num_images_in_input=2,
    use_proprio=True,
    center_crop=True,
)
processor = get_processor(cfg)
vla = get_vla(cfg)
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as f:
    obs = pickle.load(f)

action = get_vla_action(
    cfg, vla, processor, obs,
    task_label="pick up the red mug",
    action_head=action_head,
    proprio_projector=proprio_projector,
)
print("Generated action chunk:", action)
```

---

## System2 Architecture

```
Task: "put the bowl on the stove"
         │
         ▼
  ┌─────────────┐
  │ System2Agent│  ← Reasoning LLM (slow, runs every N steps)
  │  next_subgoal│
  └──────┬──────┘
         │ "move gripper toward the bowl"
         ▼
  ┌─────────────┐
  │   OpenVLA   │  ← VLA model (fast, runs every step)
  │  (OFT recipe)│
  └──────┬──────┘
         │ action chunk [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]
         ▼
      Robot
```

The `System2Agent` uses few-shot prompting to output a single next subgoal based on the current observation summary and overall task goal.

---

## Evaluation

### LIBERO Simulation

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True
```

Available pretrained checkpoints:
- `moojink/openvla-7b-oft-finetuned-libero-spatial`
- `moojink/openvla-7b-oft-finetuned-libero-object`
- `moojink/openvla-7b-oft-finetuned-libero-goal`
- `moojink/openvla-7b-oft-finetuned-libero-10`
- `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` (combined)

See [LIBERO.md](LIBERO.md) for full details.

### ALOHA Real-Robot

See [ALOHA.md](ALOHA.md) for real-robot training and evaluation instructions.

---

## Fine-Tuning

### LIBERO (Simulation)

```bash
torchrun --standalone --nnodes 1 --nproc-per-node <NUM_GPUS> vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --dataset_name libero_spatial_no_noops \
  --use_l1_regression True \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --max_steps 150005
```

### ALOHA (Real Robot)

```bash
torchrun --standalone --nnodes 1 --nproc-per-node <NUM_GPUS> vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 4 \
  --max_steps 100005
```

---

## Repository Structure

```
openvla-oft-system2/
├── experiments/robot/
│   ├── libero/
│   │   ├── system2_agent.py          # ← System2 reasoning agent (this fork)
│   │   ├── run_libero_eval.py        # LIBERO evaluation with System2 integration
│   │   └── libero_utils.py
│   ├── aloha/                        # Real-robot ALOHA scripts
│   └── openvla_utils.py              # Core VLA utilities
├── prismatic/                        # Core model architecture
│   ├── models/
│   │   ├── action_heads.py           # L1 regression & diffusion heads
│   │   ├── projectors.py             # Proprio & noisy action projectors
│   │   └── film_vit_wrapper.py       # FiLM language conditioning
│   └── vla/
│       ├── constants.py              # Robot platform configs (LIBERO/ALOHA/BRIDGE)
│       └── datasets/                 # RLDS data pipeline
├── vla-scripts/
│   ├── finetune.py                   # Main fine-tuning script
│   ├── deploy.py                     # FastAPI inference server
│   └── merge_lora_weights_and_save.py
├── test_system2_standalone.py        # ← System2 standalone test (this fork)
├── SETUP.md
├── LIBERO.md
└── ALOHA.md
```

---

## Credits and License

This repository is a fork of **[openvla/openvla](https://github.com/openvla/openvla)**.

Original work by:
- **Moo Jin Kim** (Stanford University) — moojink@cs.stanford.edu
- **Chelsea Finn** (Stanford University)
- **Percy Liang** (Stanford University)

If you use this code, please cite the original paper:

```bibtex
@article{kim25oftrecipe,
  title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
  author={Kim, Moo Jin and Pertsch, Karl and Sundaralingam, Balakumar and Davchev, Todor and Beltran-Hernandez, Camillo and Shi, Lucy Xiaoyang and Xiao, Ted and Hausman, Karol and Loquercio, Antonio and Finn, Chelsea and others},
  journal={arXiv preprint arXiv:2502.19645},
  year={2025}
}
```

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Support

For questions about the original OpenVLA-OFT framework, contact the original authors at moojink@cs.stanford.edu.

For questions specific to the System2 extension in this fork, open an issue in this repository.
