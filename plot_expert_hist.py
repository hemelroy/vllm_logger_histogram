#Plot expert usage histogram from MoE routing logs.
#Reads moe_routes.jsonl and generates expert_hist.png with:
#  - Histogram of expert usage counts
#  - Distribution statistics (top-3 experts, entropy)

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import entropy

def read_moe_log(filepath="moe_routes.jsonl"):
    """Read MoE routing log and extract metadata and routes."""
    meta = None
    routes = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            if record["type"] == "meta":
                meta = record
            elif record["type"] == "route":
                routes.append(record)
    
    return meta, routes

def analyze_expert_usage(routes):
    """Analyze expert selection patterns."""
    # Collect all expert IDs across all tokens
    all_expert_ids = []
    for route in routes:
        all_expert_ids.extend(route["topk_ids"])
    
    # Count expert usage
    expert_counts = Counter(all_expert_ids)
    
    # Get number of unique experts
    num_experts = max(all_expert_ids) + 1 if all_expert_ids else 0
    
    # Create full distribution (including unused experts)
    expert_usage = np.array([expert_counts.get(i, 0) for i in range(num_experts)])
    
    return expert_usage, expert_counts

def calculate_metrics(expert_usage):
    """Calculate distribution metrics."""
    # Normalize to get probability distribution
    total_selections = expert_usage.sum()
    if total_selections == 0:
        return {}, np.array([])
    
    prob_dist = expert_usage / total_selections
    
    # Top-3 experts
    top_3_indices = np.argsort(expert_usage)[-3:][::-1]
    top_3_experts = [
        (int(idx), int(expert_usage[idx]), float(prob_dist[idx]))
        for idx in top_3_indices
    ]
    
    # Calculate Shannon entropy (bits)
    # Filter out zero probabilities for entropy calculation
    nonzero_probs = prob_dist[prob_dist > 0]
    shannon_entropy = entropy(nonzero_probs, base=2)
    
    # Max possible entropy for uniform distribution
    num_experts = len(expert_usage)
    max_entropy = np.log2(num_experts)
    
    # Normalized entropy (0 = all load on one expert, 1 = perfectly balanced)
    normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0
    
    metrics = {
        "top_3_experts": top_3_experts,
        "shannon_entropy_bits": float(shannon_entropy),
        "max_entropy_bits": float(max_entropy),
        "normalized_entropy": float(normalized_entropy),
        "total_selections": int(total_selections),
        "num_experts": num_experts,
    }
    
    return metrics, prob_dist

def plot_histogram(expert_usage, meta, metrics, output_file="expert_hist.png"):
    """Generate and save expert usage histogram."""
    num_experts = len(expert_usage)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Raw counts
    ax1.bar(range(num_experts), expert_usage, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Expert ID', fontsize=12)
    ax1.set_ylabel('Selection Count', fontsize=12)
    ax1.set_title(f'MoE Expert Usage - {meta.get("model_id", "Unknown Model")}', 
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Highlight top-3 experts
    top_3_ids = [e[0] for e in metrics["top_3_experts"]]
    for expert_id in top_3_ids:
        ax1.bar(expert_id, expert_usage[expert_id], color='orangered', alpha=0.8)
    
    # Add text box with top-3 info
    top3_text = "Top-3 Experts:\n"
    for rank, (expert_id, count, prob) in enumerate(metrics["top_3_experts"], 1):
        top3_text += f"  #{rank}: Expert {expert_id} ({count} sel, {prob:.1%})\n"
    
    ax1.text(0.98, 0.97, top3_text.strip(), 
             transform=ax1.transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Normalized distribution
    prob_dist = expert_usage / expert_usage.sum() if expert_usage.sum() > 0 else expert_usage
    ax2.bar(range(num_experts), prob_dist, color='forestgreen', alpha=0.7)
    ax2.set_xlabel('Expert ID', fontsize=12)
    ax2.set_ylabel('Selection Probability', fontsize=12)
    ax2.set_title('Normalized Expert Distribution', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Highlight top-3 experts
    for expert_id in top_3_ids:
        ax2.bar(expert_id, prob_dist[expert_id], color='darkred', alpha=0.8)
    
    # Add entropy metric
    entropy_text = (
        f"Shannon Entropy: {metrics['shannon_entropy_bits']:.3f} bits\n"
        f"Max Entropy: {metrics['max_entropy_bits']:.3f} bits\n"
        f"Load Balance: {metrics['normalized_entropy']:.1%}\n"
        f"(1.0 = perfectly balanced)"
    )
    
    ax2.text(0.98, 0.97, entropy_text.strip(),
             transform=ax2.transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved histogram to {output_file}")
    
    return fig

def main():
    """Main analysis pipeline."""
    print("Reading MoE routing log...")
    meta, routes = read_moe_log("moe_routes.jsonl")
    
    if not routes:
        print("ERROR: No routing records found in moe_routes.jsonl")
        return
    
    print(f"✓ Loaded metadata and {len(routes)} routing records")
    print(f"  Model: {meta.get('model_id', 'Unknown')}")
    print(f"  Layers logged: {meta.get('layers_logged', [])}")
    print(f"  Top-k: {meta.get('top_k', 'Unknown')}")
    
    print("\nAnalyzing expert usage...")
    expert_usage, expert_counts = analyze_expert_usage(routes)
    metrics, prob_dist = calculate_metrics(expert_usage)
    
    print(f"\n{'='*60}")
    print(f"EXPERT USAGE ANALYSIS")
    print(f"{'='*60}")
    print(f"Total expert selections: {metrics['total_selections']}")
    print(f"Number of experts: {metrics['num_experts']}")
    print(f"Tokens processed: {len(routes)}")
    
    print(f"\nTop-3 Experts:")
    for rank, (expert_id, count, prob) in enumerate(metrics['top_3_experts'], 1):
        print(f"  #{rank}: Expert {expert_id:2d} - {count:5d} selections ({prob:6.2%})")
    
    print(f"\nDistribution Metrics:")
    print(f"  Shannon Entropy: {metrics['shannon_entropy_bits']:.4f} bits")
    print(f"  Max Entropy:     {metrics['max_entropy_bits']:.4f} bits")
    print(f"  Normalized:      {metrics['normalized_entropy']:.4f} (load balance)")
    
    # Interpretation
    if metrics['normalized_entropy'] > 0.9:
        interpretation = "near-perfect load balance across experts"
    elif metrics['normalized_entropy'] > 0.7:
        interpretation = "fairly balanced, with moderate specialization"
    elif metrics['normalized_entropy'] > 0.5:
        interpretation = "moderate imbalance, some experts are preferred"
    else:
        interpretation = "high specialization, routing heavily favors specific experts"
    
    print(f"\n  Interpretation: {interpretation}")
    print(f"{'='*60}")
    
    # Generate histogram
    print("\nGenerating histogram...")
    plot_histogram(expert_usage, meta, metrics)
    
    # Save metrics to JSON
    with open("expert_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics to expert_metrics.json")

if __name__ == "__main__":
    main()
