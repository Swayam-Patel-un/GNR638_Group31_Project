import os
import argparse
import pandas as pd
import torch
import re
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def main(test_dir):
    TEST_CSV = os.path.join(test_dir, "test.csv")
    IMAGE_DIR = os.path.join(test_dir, "images")

    print(f"Loading test csv from {TEST_CSV}")
    df = pd.read_csv(TEST_CSV)

    predictions = []

    # Local model path downloaded by setup.bash (must be absolute for newer transformers)
    MODEL_PATH = os.path.abspath("./models/Qwen2-VL-7B-Instruct")

    print("Loading model and processor...")
    
    # Load model in bfloat16 with flash attention for speed.
    # L40s (48GB) handles this easily. On Colab, remove attn_implementation.
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        print("Loaded with Flash Attention 2")
    except Exception:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        print("Loaded with default attention")

    # Auto-detect resolution based on available GPU VRAM
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3) if torch.cuda.is_available() else 0
    if gpu_mem_gb >= 40:
        max_px = 1280 * 28 * 28  # L40s (48GB)
    elif gpu_mem_gb >= 20:
        max_px = 768 * 28 * 28
    else:
        max_px = 512 * 28 * 28   # Colab T4 (16GB)
    print(f"GPU VRAM: {gpu_mem_gb:.1f} GB -> max_pixels: {max_px}")

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, 
        min_pixels=256*28*28, 
        max_pixels=max_px
    )

    print("Model loaded successfully. Starting inference...")

    for _, row in df.iterrows():
        image_name = row["image_name"]
        image_path = os.path.join(IMAGE_DIR, image_name + ".png")
        
        if not os.path.exists(image_path):
            print(f"Image {image_path} not found. Defaulting to 5 (Unanswered).")
            predictions.append({
                "id": image_name,
                "image_name": image_name,
                "option": 5
            })
            continue

        messages = [
            {
                "role": "system",
                "content": "You are an expert in deep learning. You solve multiple-choice questions accurately by reasoning step-by-step."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": """Look at this multiple-choice question image carefully. 

Step 1: Read and state the question.
Step 2: List all the options (A, B, C, D).
Step 3: Reason through each option carefully. Use standard formulas where needed, like:
  - Conv output size: floor((input + 2*padding - kernel) / stride) + 1
  - Pooling output size: floor((input - kernel) / stride) + 1
  - For nn.Linear, the bias is included by default.
Step 4: State the correct option letter.
Step 5: On the FINAL line, write ONLY: "ANSWER: X" where X is 1 for A, 2 for B, 3 for C, or 4 for D."""}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"[{image_name}] Raw output: {output_text.strip()}")

        # Try to find "ANSWER: X" pattern first (most reliable)
        match = re.search(r'ANSWER\s*:\s*([1-5])', output_text, re.IGNORECASE)
        if not match:
            # Fallback: find the last single digit 1-4 in the output
            matches = re.findall(r'\b([1-4])\b', output_text)
            if matches:
                answer = int(matches[-1])
            else:
                answer = 5
        else:
            answer = int(match.group(1))

        print(f"[{image_name}] Parsed option: {answer}")

        predictions.append({
            "id": image_name,
            "image_name": image_name,
            "option": answer
        })

    print("Inference complete. Saving to submission.csv...")
    submission = pd.DataFrame(predictions)
    
    # Check if id is present, otherwise use image_name
    if "id" not in submission.columns:
        submission["id"] = submission["image_name"]
        
    submission = submission[["id", "image_name", "option"]]
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.test_dir)
