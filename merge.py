"""
Merge LoRA adapters into the base model and save the full merged model.
Run this after training is complete.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA into base model")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="./hinglish-friend-checkpoints/final")
    parser.add_argument("--output_dir", type=str, default="./hinglish-friend-merged")
    args = parser.parse_args()

    print("Loading base model in bf16 (on CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print("Merging...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(args.output_dir)

    print("Done! Merged model saved.")


if __name__ == "__main__":
    main()
