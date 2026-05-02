# Deep Learning MCQ Solver — GNR 638 Project 2

**Team:** 22B1816, 22B1296

## Approach

We use **Qwen2-VL-7B-Instruct**, a state-of-the-art Vision-Language Model, to solve deep learning MCQs directly from images. The model reads the question image natively (no OCR needed), reasons step-by-step using chain-of-thought prompting with built-in formula references, and outputs the correct option number.

### Key Design Decisions
- **Single VLM over OCR+LLM pipeline:** OCR struggles with math notation, code blocks, and diagrams. A VLM processes the image holistically.
- **Chain-of-thought prompting:** The model reasons through each option before answering, improving accuracy on computational questions.
- **Auto VRAM detection:** Resolution automatically scales based on GPU memory (512px for 16GB, 768px for 24GB, 1280px for 48GB+).
- **Flash Attention 2:** Enabled on supported GPUs (L40s) for faster inference. Falls back gracefully on unsupported hardware.
- **Offline inference:** Model weights are downloaded during `setup.bash`. No internet is required during `inference.py`.

## Setup

```bash
bash setup.bash
```

This will:
1. Clone this repository
2. Copy `inference.py` and `requirements.txt` to the working directory
3. Create the `gnr_project_env` conda environment (Python 3.11)
4. Install all dependencies
5. Download Qwen2-VL-7B-Instruct model weights (~15GB)

## Inference

```bash
conda activate gnr_project_env
python inference.py --test_dir <absolute_path_to_test_dir>
```

This generates `submission.csv` in the current working directory.

## References
- [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) — Qwen Team, Alibaba Cloud
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
