import torch
import torch.nn.functional as F
import numpy as np
import argparse
import zlib
import os
import matplotlib.pyplot as plt
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from math import log

def get_model_stats(input_ids, model):
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Shift logits and labels to align predictions with actual next tokens
    logits = outputs.logits[0, :-1, :] 
    labels = input_ids[0, 1:] 

    #Cast logits up to float32 before math to prevent overflow/NaNs
    logits = logits.to(torch.float32)
    
    log_probs = F.log_softmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    
    token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    token_probs = probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    
    mu = (probs * log_probs).sum(-1)
    raw_sigma_square = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
    sigma = torch.clamp(raw_sigma_square, min=1e-8).sqrt()
    
    return token_log_probs, token_probs, mu, sigma

## --- METRIC 1: Min-K% ---
def score_mink(token_log_probs, k=0.2):
    num_to_keep = int(len(token_log_probs) * k)
    if num_to_keep == 0: return 0.0
    topk_log_probs, _ = torch.topk(token_log_probs, k=num_to_keep, largest=False)
    return -topk_log_probs.mean().item()

## --- METRIC 2: Min-K%++ ---
def score_mink_pp(token_log_probs, mu, sigma, k=0.2):
    mink_plus_plus_tokens = (token_log_probs - mu) / sigma
    num_to_keep = int(len(mink_plus_plus_tokens) * k)
    if num_to_keep == 0: return 0.0
    topk_scores, _ = torch.topk(mink_plus_plus_tokens, k=num_to_keep, largest=False)
    return -topk_scores.mean().item()

## --- METRIC 3: Zlib Compression Entropy ---
def score_zlib(text, token_log_probs):
    loss = -token_log_probs.mean().item()
    compressed_bytes = len(zlib.compress(text.encode('utf-8')))
    return loss / max(compressed_bytes, 1)

## --- METRIC 4: DC-PDD (On-the-fly Divergence) ---
def score_dc_pdd(labels, token_probs, fre_dis, a=0.01):
    """
    Implements DC-PDD using the on-the-fly reference distribution.
    """
    probs_array = token_probs.cpu().numpy()
    input_ids = labels.cpu().numpy()

    # SFO (Selecting First Occurrence)
    indexes = []
    current_ids = set()
    for i, input_id in enumerate(input_ids):
        if input_id not in current_ids:
            indexes.append(i)
            current_ids.add(input_id)

    if not indexes: return 0.0

    x_pro = probs_array[indexes]
    
    # Retrieve the baseline probability of the tokens from our independent distribution
    x_fre = np.array([fre_dis.get(idx, 1e-8) for idx in input_ids[indexes]])

    # Calculate Cross Entropy / Divergence
    ce = x_pro * np.log(1 / x_fre)
    
    # LUP (Limiting Upper Bound)
    ce[ce > a] = a
    
    return -np.mean(ce)


def build_independent_fre_dis(tokenizer, vocab_size, num_samples=20000):
    """
    Builds a scientifically valid proxy reference distribution by pulling a small, 
    independent chunk of text from HuggingFace (Wikitext), isolating it from the test set.
    """
    print(f"\nBuilding on-the-fly reference distribution from {num_samples} samples of Wikitext...")
    ref_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{num_samples}]")
    
    token_counts = Counter()
    total_reference_tokens = 0
    
    for item in tqdm(ref_dataset, desc="Processing Reference Tokens"):
        if not item["text"].strip(): continue
        tokens = tokenizer.encode(item["text"])
        token_counts.update(tokens)
        total_reference_tokens += len(tokens)
        
    # Convert counts to probabilities with Laplace smoothing for zero-frequency tokens
    fre_dis = {}
    for token_id in range(vocab_size):
        count = token_counts.get(token_id, 0)
        fre_dis[token_id] = (count + 1) / (total_reference_tokens + vocab_size)
        
    return fre_dis

def align_scores_for_roc(labels, scores):
    """Ensures higher scores correlate to 'memorized' (class 1)."""
    auroc = roc_auc_score(labels, scores)
    if auroc < 0.5:
        scores = [-s for s in scores]
        auroc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    return auroc, fpr, tpr

def main():
    parser = argparse.ArgumentParser(description="Evaluate memorization metrics on datasets.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the local .jsonl dataset file")
    parser.add_argument("--text_column", type=str, default="modified_text", help="The JSON key containing the text to evaluate (e.g., 'modified_text' or 'original_text')")
    parser.add_argument("--model_id", type=str, default="EleutherAI/pythia-12b-v0", help="HuggingFace model ID")
    args = parser.parse_args()

    print(f"Loading model: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    vocab_size = len(tokenizer)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map=0, torch_dtype=torch.float16 
    ).eval()

    # Build the independent reference distribution ONCE in memory
    # Note: 1,000,000 samples takes significant time. You can reduce this argument if needed for faster iteration.
    fre_dis = build_independent_fre_dis(tokenizer, vocab_size, num_samples=1000000)

    print(f"\nLoading local dataset from: {args.dataset_path}...")
    
    # Load the local JSONL file
    dataset = load_dataset("json", data_files=args.dataset_path, split="train") 
    
    results = {"mink": [], "mink_pp": [], "zlib": [], "dc_pdd": [], "custom": [], "labels": []}

    print("Evaluating metrics...")
    for item in tqdm(dataset):
        # Target the specified column (defaults to the augmented 'modified_text')
        text = item[args.text_column]
        label = item["label"]
        
        # Skip empty strings or None values
        if not text or not str(text).strip():
            continue
            
        # inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # if inputs.input_ids.shape[1] < 4: continue 
        
        # lp, p, mu, sig = get_model_stats(inputs.input_ids, model)


        try:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            if inputs.input_ids.shape[1] < 4: continue
            lp, p, mu, sig = get_model_stats(inputs.input_ids, model)

        except torch.cuda.OutOfMemoryError:
            print(f"OOM occurred, skipping...")
            torch.cuda.empty_cache()
            continue


        labels_tensor = inputs.input_ids[0, 1:]
        
        results["mink"].append(score_mink(lp))
        results["mink_pp"].append(score_mink_pp(lp, mu, sig))
        results["zlib"].append(score_zlib(text, lp))
        results["dc_pdd"].append(score_dc_pdd(labels_tensor, p, fre_dis))
        results["labels"].append(label)

    # Extract base filename for customized plot saving
    base_filename = os.path.splitext(os.path.basename(args.dataset_path))[0]
    plot_filename = f"roc_comparison_{base_filename}.png"

    # --- Plot ROC Curves ---
    print("\n--- AUROC Results & Plotting ---")
    plt.figure(figsize=(10, 8))
    
    metrics = {
        "mink": "Min-K%",
        "mink_pp": "Min-K%++",
        "zlib": "Zlib Compression",
        "dc_pdd": "DC-PDD",
        "custom": "Z-Span Density (Ours)"
    }
    
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for (key, label_name), color in zip(metrics.items(), colors):
        auroc, fpr, tpr = align_scores_for_roc(results["labels"], results[key])
        print(f"{label_name.ljust(25)}: {auroc:.4f}")
        plt.plot(fpr, tpr, lw=2, color=color, label=f'{label_name} (AUROC = {auroc:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve Comparison - {base_filename}', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(plot_filename, dpi=300)
    print(f"\nSaved ROC curve comparison plot to '{plot_filename}'")

if __name__ == "__main__":
    main()
