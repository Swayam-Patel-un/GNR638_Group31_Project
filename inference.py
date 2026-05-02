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

    # Local model path downloaded by setup.bash
    MODEL_PATH = "./models/Qwen2-VL-7B-Instruct"

    print("Loading model and processor...")
    
    # Load model in bfloat16. L40s (48GB) handles this easily.
    # On Kaggle (16GB), we use device_map="auto" to manage memory.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    # Optimization: Set resolution to model defaults for high accuracy.
    # 1280*28*28 is the recommended high-res setting for Qwen2-VL.
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, 
        min_pixels=256*28*28, 
        max_pixels=1280*28*28 
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
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Analyze this deep learning multiple-choice question image. Identify the correct option among A, B, C, and D. Output ONLY the corresponding number: 1 for A, 2 for B, 3 for C, or 4 for D. If you are uncertain or the image is unclear, output 5."}
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
            generated_ids = model.generate(**inputs, max_new_tokens=10)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"[{image_name}] Raw output: {output_text.strip()}")

        match = re.search(r'\b([1-5])\b', output_text)
        if match:
            answer = int(match.group(1))
        else:
            answer = 5 

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
