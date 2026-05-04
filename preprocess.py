"""
Preprocess Hinglish-Everyday-Conversations-1M dataset for Llama-3.1-8B-Instruct finetuning.
Converts input/output pairs into Llama chat format and saves as a HuggingFace dataset.
"""

import argparse
from datasets import load_dataset, DatasetDict

SYSTEM_PROMPT = (
    "Tu ek friendly Indian dost hai jo Hinglish mein baat karta hai. "
    "Casual, fun, aur relatable tone mein reply kar. "
    "Short aur natural responses de, jaise koi close friend baat kar raha ho."
)


def format_to_chat(example):
    """Convert input/output pair to Llama-3.1 chat message format."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return {"messages": messages}


def filter_by_length(example, min_output_chars=10, max_output_chars=500):
    """Filter out very short or very long responses."""
    output_len = len(example["output"])
    return min_output_chars <= output_len <= max_output_chars


def main():
    parser = argparse.ArgumentParser(description="Preprocess Hinglish dataset")
    parser.add_argument("--num_samples", type=int, default=300_000, help="Number of samples to use")
    parser.add_argument("--min_output_chars", type=int, default=10)
    parser.add_argument("--max_output_chars", type=int, default=500)
    parser.add_argument("--eval_ratio", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default="./processed-data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("Abhishekcr448/Hinglish-Everyday-Conversations-1M", split="train")
    print(f"Original dataset size: {len(dataset)}")

    # Filter by output length
    dataset = dataset.filter(
        lambda x: filter_by_length(x, args.min_output_chars, args.max_output_chars),
        desc="Filtering by length",
    )
    print(f"After length filtering: {len(dataset)}")

    # Shuffle and take subset
    dataset = dataset.shuffle(seed=args.seed)
    if len(dataset) > args.num_samples:
        dataset = dataset.select(range(args.num_samples))
    print(f"Using {len(dataset)} samples")

    # Convert to chat format
    dataset = dataset.map(format_to_chat, desc="Formatting to chat")

    # Remove original columns, keep only 'messages'
    dataset = dataset.remove_columns(["input", "output"])

    # Split into train/eval
    split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    final_dataset = DatasetDict({
        "train": split["train"],
        "eval": split["test"],
    })

    print(f"Train: {len(final_dataset['train'])}, Eval: {len(final_dataset['eval'])}")

    # Save
    final_dataset.save_to_disk(args.output_dir)
    print(f"Saved to {args.output_dir}")

    # Print a sample
    print("\n--- Sample conversation ---")
    sample = final_dataset["train"][0]["messages"]
    for msg in sample:
        print(f"[{msg['role']}]: {msg['content']}")


if __name__ == "__main__":
    main()
