import json
import matplotlib.pyplot as plt
import numpy as np

# Load JSON data
with open('../data/eval/eval_ragas.json', 'r') as f:
    rag_data = json.load(f)

with open('../data/eval/medical/big/eval.json', 'r') as f:
    eval_data = json.load(f)

# Create path-indexed dictionary for quick lookup
eval_scores = {entry['path']: entry['average_score']*100/5 for entry in eval_data}

# Extract data for plotting
paths = []
faithfulness = []
relevancy = []
combined_scores = []

for entry in rag_data:
    path_key = entry['path']
    base_name = path_key.split('/')[-1].replace('_answers.json', '')
    paths.append(base_name)
    faithfulness.append(entry['average_score_faithfullness']*100)
    relevancy.append(entry['average_score_answer_relevancy']*100)
    combined_scores.append(eval_scores[path_key])

# Visualization parameters
x = np.arange(len(paths))  # Category positions
total_width = 0.8         # Total width for all bars in a group
bar_width = total_width / 3  # Width for individual bars

fig, ax = plt.subplots(figsize=(14, 8))
rects1 = ax.bar(x - bar_width, faithfulness, bar_width,
                label='RAGAS Faithfulness', color='#1f77b4', edgecolor='white')
rects2 = ax.bar(x, relevancy, bar_width,
                label='RAGAS Answer Relevancy', color='#ff7f0e', edgecolor='white')
rects3 = ax.bar(x + bar_width, combined_scores, bar_width,
                label='LLM as judge prompting', color='#2ca02c', edgecolor='white')

# Formatting enhancements
ax.set_title('Multi-Metric RAG Configuration Comparison', fontsize=16, pad=20)
ax.set_xlabel('RAG Methods', fontsize=12)
ax.set_ylabel('Normalized Scores', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([p.capitalize() for p in paths], fontsize=11)
ax.legend(title='Metric Type', fontsize=10, title_fontsize=11)
ax.grid(True, linestyle=':', alpha=0.7)
ax.set_axisbelow(True)

# Dynamic annotation positioning
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3",
                            fc="white", ec="grey", lw=0.5))

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# Adjust layout and save
plt.tight_layout()
plt.savefig('multi_metric_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


plt.tight_layout()
plt.savefig('rag_evaluation_comparison_ragas.png', dpi=300)
plt.show()