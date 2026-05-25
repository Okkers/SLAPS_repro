import numpy as np 

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def aggregate_metrics(all_accuracies,args):
    """
    Aggregate metrics across 10 runs.
    """
    if len(all_accuracies) > 0:
        mean_acc = np.mean(all_accuracies) * 100
        std_acc = np.std(all_accuracies) * 100

        with open("results_std_mean.txt", "a") as f:
            f.write(f"Experiment Setup: {args.model_id}_{args.model}_{args.dataset}\n")
            f.write(f"Accuracies over {args.itr} runs: {[round(a * 100, 2) for a in all_accuracies]}\n")
            f.write(f"Final Accuracy: {mean_acc:.2f} ± {std_acc:.2f}\n")
            f.write("-" * 40 + "\n\n")