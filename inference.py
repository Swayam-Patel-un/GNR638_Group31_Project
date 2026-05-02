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
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
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
Step 3: Reason through each option carefully. Use these standard formulas where needed:

CONVOLUTION & POOLING:
  - Conv output: floor((input + 2*padding - kernel) / stride) + 1
  - Transposed Conv output: (input - 1)*stride - 2*padding + kernel + output_padding
  - Pooling output: floor((input - kernel) / stride) + 1
  - Receptive field: R_new = R_old + (kernel - 1) * jump
  - #Params in Conv2d: (kernel_h * kernel_w * in_channels + 1) * out_channels (with bias)

FULLY CONNECTED:
  - nn.Linear(in, out) has weight shape (out, in) and bias shape (out)
  - #Params: in * out + out (with bias)
  - nn.Linear includes bias by default

BATCH NORM:
  - BatchNorm has 2 learnable params per channel (gamma, beta) + 2 running stats (mean, var)
  - Applied BEFORE or AFTER activation (common: Conv -> BN -> ReLU)

ACTIVATIONS:
  - ReLU(x) = max(0, x), derivative = 1 if x > 0 else 0
  - Sigmoid(x) = 1/(1+e^(-x)), range (0,1)
  - Tanh(x) = (e^x - e^(-x))/(e^x + e^(-x)), range (-1,1)
  - Softmax(x_i) = e^(x_i) / sum(e^(x_j))
  - LeakyReLU(x) = x if x > 0, else alpha*x

LOSS FUNCTIONS:
  - Cross-entropy: -sum(y * log(p))
  - Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
  - MSE: mean((y - y_hat)^2)

OPTIMIZERS:
  - SGD: w = w - lr * grad
  - Momentum: v = beta*v + grad, w = w - lr*v
  - Adam: combines momentum + RMSprop, uses m (1st moment) and v (2nd moment)

REGULARIZATION:
  - Dropout: randomly zeros elements with probability p, scales by 1/(1-p) during training
  - L1 adds |w| to loss, L2 adds w^2 to loss (weight decay)

RNN/LSTM/GRU:
  - RNN hidden: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
  - LSTM has 3 gates: forget, input, output + cell state
  - GRU has 2 gates: reset, update
  - Bidirectional doubles hidden size output

TRANSFORMER:
  - Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
  - Multi-head: concat(head_1,...,head_h) * W_O
  - Positional encoding added to embeddings
  - Self-attention complexity: O(n^2 * d)

GENERAL:
  - Gradient vanishing: common with sigmoid/tanh in deep networks
  - ResNet skip connections: output = F(x) + x
  - 1x1 convolution: changes channel dimension without changing spatial size

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
            "image_name": image_name,
            "option": answer
        })

    print("Inference complete. Saving to submission.csv...")
    submission = pd.DataFrame(predictions)
    submission = submission[["image_name", "option"]]
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.test_dir)
