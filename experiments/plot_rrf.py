import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    csv_path = os.path.join(base_dir, "rrf_results.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}. Please run benchmark_rrf_sensitivity.py first.")
        return
        
    plot_path = os.path.join(base_dir, "rrf_plot.png")

    # Read the data
    df = pd.read_csv(csv_path)

    # Set up the plot style
    plt.figure(figsize=(10, 6))
    
    # We plot NDCG@10 
    plt.plot(df["k"], df["NDCG@10"], marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)
    
    # Find the maximum point for annotation
    max_idx = df["NDCG@10"].idxmax()
    max_k = df["k"][max_idx]
    max_ndcg = df["NDCG@10"][max_idx]
    
    # Highlight the maximum point
    plt.plot(max_k, max_ndcg, marker='o', color='red', markersize=10)
    plt.annotate(f"Optimal k={max_k}\nNDCG={max_ndcg:.4f}", 
                 (max_k, max_ndcg), 
                 textcoords="offset points", 
                 xytext=(20, 10), 
                 ha='center',
                 fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.4", fc="yellow", ec="black", lw=1, alpha=0.5))

    plt.title('RRF Parameter Sensitivity (k vs NDCG@10)', fontsize=16, pad=15)
    plt.xlabel('Smoothing Constant (k)', fontsize=14)
    plt.ylabel('NDCG@10', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved successfully to {plot_path}")

if __name__ == "__main__":
    main()
