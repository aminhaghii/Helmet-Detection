import matplotlib

matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import base64

# Load the training results
results_path = "c:/Users/aminh/OneDrive/Desktop/Projects/HSE_Vision/models/trained/helmet_detection_20250727_152811/results.csv"
results_df = pd.read_csv(results_path)
results_df.columns = results_df.columns.str.strip()

# Define output directory and create it if it doesn't exist
output_dir = (
    "c:/Users/aminh/OneDrive/Desktop/Projects/HSE_Vision/outputs/visualizations"
)
os.makedirs(output_dir, exist_ok=True)

# Set plot style
sns.set_style("whitegrid")

# Plot Training & Validation Loss
fig1 = plt.figure(figsize=(12, 8))
plt.plot(results_df["epoch"], results_df["train/box_loss"], label="Train Box Loss")
plt.plot(results_df["epoch"], results_df["val/box_loss"], label="Val Box Loss")
plt.plot(results_df["epoch"], results_df["train/cls_loss"], label="Train Class Loss")
plt.plot(results_df["epoch"], results_df["val/cls_loss"], label="Val Class Loss")
plt.plot(results_df["epoch"], results_df["train/dfl_loss"], label="Train DFL Loss")
plt.plot(results_df["epoch"], results_df["val/dfl_loss"], label="Val DFL Loss")
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
save_path1 = os.path.join(output_dir, "loss_curves.png")
buf = io.BytesIO()
plt.savefig(buf, format="png")
buf.seek(0)
img_str = base64.b64encode(buf.read()).decode("utf-8")
print(f"LOSS_CURVE_BASE64:{img_str}")
plt.close(fig1)

# Plot Precision, Recall, and mAP@.50
fig2 = plt.figure(figsize=(12, 8))
plt.plot(results_df["epoch"], results_df["metrics/precision(B)"], label="Precision")
plt.plot(results_df["epoch"], results_df["metrics/recall(B)"], label="Recall")
plt.plot(results_df["epoch"], results_df["metrics/mAP50(B)"], label="mAP@.50")
plt.title("Precision, Recall, and mAP@.50")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
save_path2 = os.path.join(output_dir, "precision_recall_map50.png")
buf = io.BytesIO()
plt.savefig(buf, format="png")
buf.seek(0)
img_str = base64.b64encode(buf.read()).decode("utf-8")
print(f"METRICS_CURVE_BASE64:{img_str}")
plt.close(fig2)

# Plot mAP@.50-95
fig3 = plt.figure(figsize=(12, 8))
plt.plot(results_df["epoch"], results_df["metrics/mAP50-95(B)"], label="mAP@.50-95")
plt.title("mAP@.50-95")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
save_path3 = os.path.join(output_dir, "map50-95.png")
buf = io.BytesIO()
plt.savefig(buf, format="png")
buf.seek(0)
img_str = base64.b64encode(buf.read()).decode("utf-8")
print(f"MAP_CURVE_BASE64:{img_str}")
plt.close(fig3)

print("Plots generated successfully!")
