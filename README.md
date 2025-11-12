# Magistral Fine-Tuning (Mult-IT)
This repository contains tools and helper code to fine-tune Magistral models with support for "thinking" (chain-of-thought) and mixed thinking/non-thinking training modes. 
It includes a compact training wrapper, configuration class, and dataset utilities to prepare and run supervised fine-tuning with QLoRA and TRL/SFT workflows.

# Requirements
- Python: 3.10 to 3.12
- CUDA: 21.1+ (install a matching PyTorch wheel for your CUDA runtime)

Note: adjust the PyTorch wheel URL or package versions to match your local CUDA runtime (the examples below use the cu121 wheel index).

# Stack

- [PyTorch](https://pytorch.org/): Core deep learning library used for model execution and GPU acceleration.
- [Transformers](https://github.com/huggingface/transformers): Model and tokenizer loading from Hugging Face Hub.
- [PEFT](https://github.com/huggingface/peft): Parameter-Efficient Fine-Tuning (LoRA) utilities.
- [Accelerate](https://github.com/huggingface/accelerate): Device and distributed training utilities.
- [TRL (trl)](https://github.com/lvwerra/trl): Training utilities for SFT / policy learning.
- [datasets](https://github.com/huggingface/datasets): Dataset utilities and I/O.
- [scikit-learn](https://scikit-learn.org/): Evaluation and metrics.
- [tqdm](https://github.com/tqdm/tqdm): Progress bars.
- [pandas](https://pandas.pydata.org/): Data inspection and tabulation.

# Set Up

Two supported ways to set up the environment: pip (virtualenv) and Poetry. Pick the one you prefer.

Important: install a matching NVIDIA driver and CUDA toolkit on your system before running Poetry so that GPU-enabled PyTorch can be installed and used.

General CUDA install steps (follow the official NVIDIA instructions for your OS and desired CUDA version):

- Install NVIDIA GPU driver (check `nvidia-smi` after install).
- Install the CUDA toolkit/version that matches the PyTorch build you intend to use (e.g. CUDA 12.1 for cu121).
- Verify installation:

```bash
nvidia-smi
nvcc --version
```

## 1. Using pip + virtualenv

Create and activate a virtual environment, then install packages. Adjust the PyTorch index URL to match your CUDA version if needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install PyTorch wheels from the official PyTorch wheel index (example: CUDA 12.1 / cu121)
```
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining Python dependencies
```
pip install transformers pandas numpy tqdm python-dotenv peft bitsandbytes jupyter datasets
pip install numpy transformers peft accelerate scikit-learn python-dotenv jupyter trl python-dotenv tqdm datasets pandas tiktoken protobuf mistral-common
```

# References
- [Mult-IT: Multiple Choice Questions on Multiple Topics in Italian:A CALAMITA Challenge](https://aclanthology.org/2024.clicit-1.131.pdf)