"""
Test inference with the finetuned Hinglish chatbot model.
Can load either the LoRA adapter or the merged model.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

SYSTEM_PROMPT = (
    "Tu ek friendly Indian dost hai jo Hinglish mein baat karta hai. "
    "Casual, fun, aur relatable tone mein reply kar. "
    "Short aur natural responses de, jaise koi close friend baat kar raha ho."
)

TEST_PROMPTS = [
    "kya yaar, traffic mein stuck ho gaya",
    "aaj weekend hai, kya plan hai?",
    "bhai bohot bura mood hai aaj",
    "new phone liya maine, guess kar kitne ka",
    "chai peene chalein?",
    "office mein boss ne phir se daant diya",
    "bro movie dekhne chalein kya aaj raat?",
    "yaar paisa khatam ho gaya mahine ke end mein",
]


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.use_adapter:
        # Load base model in 4-bit + LoRA adapter
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(model, args.model_path)
    else:
        # Load merged model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, user_input, max_new_tokens=256):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Test Hinglish chatbot")
    parser.add_argument("--model_path", type=str, default="./hinglish-friend-merged",
                        help="Path to merged model or LoRA adapter")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Base model (only needed if --use_adapter)")
    parser.add_argument("--use_adapter", action="store_true",
                        help="Load as LoRA adapter instead of merged model")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive chat mode")
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load_model(args)
    print("Model loaded!\n")

    if args.interactive:
        print("Interactive mode. Type 'quit' to exit.\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            response = generate_response(model, tokenizer, user_input)
            print(f"Bot: {response}\n")
    else:
        print("Running test prompts...\n")
        for prompt in TEST_PROMPTS:
            response = generate_response(model, tokenizer, prompt)
            print(f"User: {prompt}")
            print(f"Bot:  {response}")
            print("-" * 60)


if __name__ == "__main__":
    main()
