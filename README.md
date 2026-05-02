# Deep Learning MCQ Solver (Qwen2-VL)

This project solves Multiple Choice Questions on Deep Learning from images using the state-of-the-art **Qwen2-VL-7B-Instruct** Vision-Language Model.

## Architecture Highlights
- **Vision-Language Model (VLM):** We abandoned the brittle OCR+LLM pipeline in favor of a single unified VLM. Qwen2-VL processes both the image and text simultaneously. This completely solves the problem of misreading math equations, formatting, and complex diagrams which traditional OCR fails at.
- **Offline Inference:** The model is downloaded during the `setup.bash` phase. No internet connection is required during inference, adhering strictly to the guidelines.
- **Fast & Memory Efficient:** Running in `bfloat16`, the 7B model easily fits in the provided 48GB VRAM (using ~14GB) and generates answers rapidly.

## Setup Instructions

Simply run the provided setup script. It will create the `gnr_project_env` conda environment, install dependencies, and download the necessary models from HuggingFace to the local directory.

```bash
bash setup.bash
```

## Inference

Run the inference script as specified in the guidelines:

```bash
conda activate gnr_project_env
python inference.py --test_dir <absolute_path_to_test_dir>
```

This will read the images from the test directory and generate a `submission.csv` in the current working directory.
