import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

data = [
    (0, 1, 1.0),
    (1, 2, 0.9),
    (2, 1, 0.8),
    (3, 1, 0.7),
    (4, 2, 0.6),
    (5, 1, 0.5),
    (6, 2, 0.4),
    (7, 2, 0.3),
    (8, 1, 0.2),
    (9, 2, 0.1)
]

true_labels = np.array([label for _, label, _ in data])
scores = np.array([score for _, _, score in data])

binary_labels = (true_labels == 1).astype(int)
precision, recall, _ = precision_recall_curve(binary_labels, scores)
auc_pr = auc(recall, precision)
ap = average_precision_score(binary_labels, scores)

print(f"AUC-PR: {auc_pr:.4f}")
print(f"AP: {ap:.4f}")

plt.figure()
plt.plot(recall, precision, marker='.', label=f'AUC-PR = {auc_pr:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('pr.png')
plt.show()
