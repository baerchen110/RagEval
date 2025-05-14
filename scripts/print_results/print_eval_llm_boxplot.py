import json
import matplotlib.pyplot as plt
import numpy as np

# Load JSON data
with open('../../data/eval/medical/multi/eval.json', 'r') as f:
    data = json.load(f)

# Extract document names and scores
max_score = 5
doc_names = [entry['path'].split('/')[-1].replace('_answers.json', '') for entry in data]
scores_percent = {name: [(score/max_score)*100 for score in doc['total_scores']]
                 for name, doc in zip(doc_names, data)}

# Create figure
plt.figure(figsize=(12, 7))
boxes = plt.boxplot(scores_percent.values(), patch_artist=True, tick_labels=scores_percent.keys())

# Customize boxplot appearance
for box in boxes['boxes']:
    box.set_facecolor('#1f77b4')
    box.set_alpha(0.5)

# Add statistical markers
for i, (doc, values) in enumerate(scores_percent.items()):
    # Calculate statistics
    min_val = np.min(values)
    max_val = np.max(values)
    avg_val = np.mean(values)

    # Plot markers
    plt.plot(i + 1, min_val, 'r^', markersize=8, label='Min' if i == 0 else "")
    plt.plot(i + 1, max_val, 'rv', markersize=8, label='Max' if i == 0 else "")
    plt.plot(i + 1, avg_val, 'go', markersize=8, label='Mean' if i == 0 else "")

# Add labels and legend
plt.title('RAG Benchmark LLM as a judge', fontsize=14)
plt.xlabel('RAG methods', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Statistics', loc='upper right')

plt.show()
