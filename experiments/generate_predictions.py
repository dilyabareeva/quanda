import argparse
import random

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def create_prompt_dataset(
    dataset_name,
    tokenizer_name,
    num_prompts,
    seed=42,
):
    random.seed(seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"Loading dataset '{dataset_name}'...")
    dataset = load_dataset(dataset_name, split="train")

    # Select prompts
    if num_prompts < len(dataset):
        print(
            f"Selecting {num_prompts} prompts from the dataset (total: {len(dataset)})."
        )
        random.seed(seed)
        indices = random.sample(range(len(dataset)), num_prompts)
        dataset = dataset.select(indices)
    else:
        print(f"Using the full dataset ({len(dataset)} prompts).")
        dataset = dataset

    # Create dataset
    prompts = []
    answers = []
    alt_answers = []

    for entry in dataset:
        prompt = entry["prompt"].strip()
        answer = entry["answer"][0].strip()
        alt_answer_list = entry.get("alt_answers", [])
        if not isinstance(alt_answer_list, list):
            alt_answer_list = [str(alt_answer_list)]
        else:
            alt_answer_list = [str(a).strip() for a in alt_answer_list]

        prompts.append(prompt)
        answers.append(answer)
        alt_answers.append(alt_answer_list)

    prompt_dataset = Dataset.from_dict(
        {
            "prompt": prompts,
            "answer": answers,
            "alt_answers": alt_answers,
        }
    )

    return prompt_dataset, tokenizer


def generate_predictions(model, tokenizer, prompt_dataset):
    model.eval()
    correct_count = 0
    total_prompts = len(prompt_dataset)

    print("\n=== Predictions ===\n")
    for i in range(total_prompts):
        prompt_text = prompt_dataset[i]["prompt"]
        target = prompt_dataset[i]["answer"]
        alt_answers = prompt_dataset[i]["alt_answers"]

        # Encode the prompt
        encoding = tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Generate prediction
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )[0]

        # Decode the generated sequence
        decoded = tokenizer.decode(output_ids)

        # Extract only the predicted part
        predicted_part = decoded[len(prompt_text) :].strip()

        # Check if answer is correct
        prediction_lower = predicted_part.lower()
        answers = [a.lower() for a in alt_answers] + [target.lower()]

        found_match = False
        for answer in answers:
            if answer in prediction_lower:
                found_match = True
                break

        match_symbol = "\u2705" if found_match else "\u274c"
        if found_match:
            correct_count += 1

        # Replace newlines with spaces before printing
        predicted_part = predicted_part.replace("\n", " ")

        print(f"Prompt   [{i+1}]: {prompt_text}")
        print(f"Target      → {target}")
        print(f"Prediction  → {predicted_part} {match_symbol}")
        print("-" * 60)

    # Calculate the accuracy
    accuracy = (correct_count / total_prompts) * 100
    print("\n=== Summary ===")
    print(f"Correct Predictions: {correct_count} / {total_prompts}")
    print(f"Accuracy: {accuracy:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions for T-REx prompts."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Name of the Hugging Face model to use (default: gpt2).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="quanda-bench-test/trex-subset",
        help="Name of the Hugging Face dataset to use (default: quanda-bench-test/trex-subset).",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=1000,
        help="Number of prompts to sample from the data file (default: 100).",
    )
    args = parser.parse_args()

    print(f"Loading model '{args.model_name}'...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    prompt_dataset, tokenizer = create_prompt_dataset(
        dataset_name=args.dataset_name,
        tokenizer_name=args.model_name,
        num_prompts=args.num_prompts,
    )

    generate_predictions(model, tokenizer, prompt_dataset)


if __name__ == "__main__":
    main()
