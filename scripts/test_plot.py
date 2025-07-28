import matplotlib.pyplot as plt
import os

output_dir = (
    "c:/Users/aminh/OneDrive/Desktop/Projects/HSE_Vision/outputs/visualizations"
)
os.makedirs(output_dir, exist_ok=True)

plt.figure()
plt.plot([1, 2, 3], [1, 4, 9])
plt.title("Test Plot")
save_path = os.path.join(output_dir, "test_plot.png")
print(f"Saving plot to: {save_path}")
plt.savefig(save_path)
plt.close()

print("Test plot script finished.")

# Verify file exists
if os.path.exists(save_path):
    print("File created successfully.")
else:
    print("File was not created.")
