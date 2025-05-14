import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the JSON data
json_path = '../../data/eval/law/en/bg3/eval_ragas.json'
with open(json_path, 'r') as f:
    data = json.load(f)

# Extract document names
doc_names = [os.path.basename(entry['path']).replace('_answers.json', '').replace('.json', '') for entry in data]

# Define metrics and colors
metrics = [
    ('faithfulness', '#1f77b4'),
    ('answer_relevancy', '#ff7f0e'),
    ('context_precision', '#2ca02c'),
    ('context_recall', '#d62728')
]
metric_names = [m[0] for m in metrics]
colors = [m[1] for m in metrics]

# Extract average scores for each metric and document
avg_scores = {m: [] for m in metric_names}
for entry in data:
    for m in metric_names:
        avg_key = f'average_score_{m}'
        avg = entry.get(avg_key, None)
        if avg is not None:
            avg_scores[m].append(avg * 100)  # convert to percentage
        else:
            avg_scores[m].append(np.nan)

# Prepare data for plotting
x = np.arange(len(doc_names))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for each metric
for i, m in enumerate(metric_names):
    ax.bar(x + i*width, avg_scores[m], width, label=m.replace('_', ' ').capitalize(), color=colors[i])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Score (%)')
ax.set_title('Average Scores Eval Metric and RAG Method')
ax.set_xticks(x + width * (len(metrics)-1) / 2)
ax.set_xticklabels(doc_names, rotation=30, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
