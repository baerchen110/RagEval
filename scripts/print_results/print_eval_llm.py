import json
import matplotlib.pyplot as plt
import numpy as np

# Load JSON data
with open('../../data/eval/medical/big/eval.json', 'r') as f:
    data = json.load(f)

# Extract data for plotting
paths = [entry['path'].split('/')[-1].replace('_answers.json', '') for entry in data]
faithfulness = [entry['average_score']*100/5 for entry in data]

# Plot configuration
x = np.arange(len(paths))  # label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
rects = ax.bar(x - width/2, faithfulness, width, label='Faithfulness', color='#1f77b4')

# Add labels and title
ax.set_ylabel('Scores', fontsize=12)
ax.set_title('RAG Evaluation Metrics Comparison LLM as judge', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels([p.capitalize() for p in paths], fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.6)

# Add value labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9)
autolabel(rects)


plt.tight_layout()
plt.savefig('rag_evaluation_comparison_llm.png', dpi=300)
plt.show()