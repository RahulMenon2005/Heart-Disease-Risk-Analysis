import numpy as np
import matplotlib.pyplot as plt
from src.model import get_performance_dataframe

def plot_model_comparison(models, X_test, y_test):

    results_df = get_performance_dataframe(models, X_test, y_test)
  
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    models_list = results_df["Model"]
    x = np.arange(len(models_list))
    width = 0.18

    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, results_df[metric], width, label=metric)

    plt.xticks(x + width * 1.5, models_list, rotation=15, ha='right')
    plt.ylabel("Score (0.0 - 1.0)")
    plt.ylim(0, 1.1)
    plt.title("Dynamic Model Performance Comparison", fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()