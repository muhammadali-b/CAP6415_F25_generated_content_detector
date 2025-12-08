import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Confusion matrix values ---
cm = np.array([
    [7047, 501],
    [471, 7033]
])

class_names = ["real", "ai"]

# --- Plot ---
plt.figure(figsize=(6, 5))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

backend_dir = Path(__file__).resolve().parents[1]
project_root = backend_dir.parent

results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)

output_path = results_dir / "confusion_matrix.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Saved confusion matrix to {output_path}")
