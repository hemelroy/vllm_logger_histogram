# Run vLLM generation on GSM8K prompts with MoE expert logging.
# This script generates text using the Qwen1.5-MoE-A2.7B-Chat model and
# measures wall-clock time.

import os
import json
import time
import random

from vllm import LLM, SamplingParams

# Set random seed for reproducibility
SEED = 1234
random.seed(SEED)

# Load prompts
print("Loading prompts...")
with open("prompts.txt", "r", encoding="utf-8") as f:
    prompts = f.read().split("\n\n---\n\n")

print(f"Loaded {len(prompts)} prompts")

# Configure sampling parameters
sp = SamplingParams(
    temperature=0.0,  # Greedy decoding
    max_tokens=128,   # Generation cap
    seed=SEED,
)

# Initialize vLLM engine
print("Initializing vLLM engine...")
print("Model: Qwen/Qwen1.5-MoE-A2.7B-Chat")

llm = LLM(
    model="Qwen/Qwen1.5-MoE-A2.7B-Chat",
    max_model_len=512,  # Keep it small for this experiment
    trust_remote_code=True,
)

# Check if logging is enabled
log_mode = "WITH LOGGING" if os.environ.get("VLLM_LOG_MOE") else "NO LOGGING"
print(f"\nRunning in {log_mode} mode")

# Generate
print("Generating responses...")
t0 = time.time()
outputs = llm.generate(prompts, sp)
t1 = time.time()

# Calculate statistics
wall_time_sec = t1 - t0
total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

print(f"\n{'='*60}")
print(f"Generation complete!")
print(f"  Wall time: {wall_time_sec:.2f} seconds")
print(f"  Tokens generated: {total_tokens}")
print(f"  Throughput: {total_tokens / wall_time_sec:.2f} tokens/sec")
print(f"{'='*60}")

# Save or update timing results
timing_file = "timing.json"
if os.path.exists(timing_file):
    with open(timing_file, "r") as f:
        timing_data = json.load(f)
else:
    timing_data = {}

# Update with current run
if os.environ.get("VLLM_LOG_MOE"):
    timing_data["log"] = {
        "wall_time_sec": wall_time_sec,
        "tokens_generated": total_tokens,
    }
    print(f"\n✓ Updated timing.json with 'log' results")
else:
    timing_data["no_log"] = {
        "wall_time_sec": wall_time_sec,
        "tokens_generated": total_tokens,
    }
    print(f"\n✓ Updated timing.json with 'no_log' results")

# Save timing data
with open(timing_file, "w") as f:
    json.dump(timing_data, f, indent=2)

# Calculate overhead if both runs are available
if "no_log" in timing_data and "log" in timing_data:
    overhead_sec = timing_data["log"]["wall_time_sec"] - timing_data["no_log"]["wall_time_sec"]
    overhead_pct = (overhead_sec / timing_data["no_log"]["wall_time_sec"]) * 100
    print(f"\nLogging overhead: {overhead_sec:.3f}s ({overhead_pct:.2f}%)")

print(f"\n✓ Saved {len(outputs)} outputs")
print(f"✓ Example output (first prompt):")
print(f"  Prompt: {prompts[0][:80]}...")
print(f"  Response: {outputs[0].outputs[0].text[:150]}...")
