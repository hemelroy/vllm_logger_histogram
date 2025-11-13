# Generate prompts from GSM8K dataset for MoE expert logging experiment. 
# Loads the first 25 questions from the GSM8K test split
# and saves to prompts.txt

from datasets import load_dataset

# Load GSM8K test split (MIT licensed)
print("Loading GSM8K dataset...")
ds = load_dataset("openai/gsm8k", "main", split="test")

# Extract first 25 questions
print("Extracting first 25 questions...")
prompts = [ex["question"] for ex in ds.select(range(25))]

# Write to file with separator
output_file = "prompts.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n\n---\n\n".join(prompts))

print(f"✓ Saved {len(prompts)} prompts to {output_file}")
print(f"  Total characters: {sum(len(p) for p in prompts)}")
