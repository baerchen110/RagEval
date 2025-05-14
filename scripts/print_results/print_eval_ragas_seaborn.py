import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DEFAULT_JSON_PATH = '../../data/archive/bge_e5.json'
MAX_SCORE = 1.0  # Used to normalize scores to percentage

# Define metrics and colors
METRICS = [
    ('faithfulness', '#1f77b4'),
    ('answer_relevancy', '#ff7f0e'),
    ('context_precision', '#2ca02c'),
    ('context_recall', '#d62728')
]

def load_data(json_path):
    """Load JSON data from file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_doc_names(data):
    """Extract document names from data."""
    return [
        os.path.basename(entry['path']).replace('_answers.json', '').replace('.json', '')
        for entry in data
    ]

def prepare_plot_data(data, metric_names):
    """Prepare data for plotting."""
    plot_data = {m: [] for m in metric_names}
    for entry in data:
        for m in metric_names:
            key = f'total_scores_{m}'
            scores = entry.get(key, [])
            if not scores:
                plot_data[m].append([])
            else:
                percent_scores = [(score / MAX_SCORE) * 100 for score in scores]
                plot_data[m].append(percent_scores)
    return plot_data

def print_summary_stats(data, metric_names, doc_names):
    print('\nAverage Scores (%):')
    for i, entry in enumerate(data):
        print(f" {doc_names[i]}:")
        for m in metric_names:
            avg_key = f'average_score_{m}'
            avg = entry.get(avg_key, None)
            if avg is not None:
                print(f"   {m:18}: {avg * 100:.1f}")
            else:
                print(f"   {m:18}: N/A")

def plot_violinplots(plot_data, doc_names, metrics):
    metric_names = [m[0] for m in metrics]
    colors = [m[1] for m in metrics]
    n_docs = len(doc_names)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_docs, figsize=(4 * n_docs, 7), sharey=True)
    if n_docs == 1:
        axes = [axes]

    for doc_idx, ax in enumerate(axes):
        all_scores = []
        all_metrics = []
        palette = []
        for m_idx, m in enumerate(metric_names):
            scores = plot_data[m][doc_idx]
            if not scores or len(set(scores)) <= 1:
                # If empty or constant, plot a single point
                y = scores[0] if scores else 0
                ax.scatter([m], [y], color=colors[m_idx], s=80, label=f"{m} (single value)")
            else:
                all_scores.extend(scores)
                all_metrics.extend([m] * len(scores))
                palette.append(colors[m_idx])

        # Only plot if there is more than one unique value
        if all_scores:
            sns.violinplot(
                x=all_metrics,
                y=all_scores,
                ax=ax,
                palette=colors,
                inner='box',
                cut=0,
                linewidth=1.2
            )
            sns.stripplot(
                x=all_metrics,
                y=all_scores,
                ax=ax,
                color='k',
                size=3,
                jitter=0.2,
                dodge=False,
                alpha=0.5
            )

        ax.set_title(doc_names[doc_idx])
        ax.set_xlabel('')
        if doc_idx == 0:
            ax.set_ylabel('Score (%)')
        else:
            ax.set_ylabel('')
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.set_xticklabels([m.replace('_', ' ').capitalize() for m in metric_names], rotation=30, ha='right')

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=name.replace('_', ' ').capitalize(), markersize=10)
        for name, color in metrics
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    plt.suptitle('RAG Evaluation Metrics by Document', fontsize=16, y=1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def main():
    # Allow optional JSON path argument
    json_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSON_PATH
    data = load_data(json_path)
    metric_names = [m[0] for m in METRICS]
    doc_names = extract_doc_names(data)
    plot_data = prepare_plot_data(data, metric_names)
    print_summary_stats(data, metric_names, doc_names)
    plot_violinplots(plot_data, doc_names, METRICS)

if __name__ == "__main__":
    main()
