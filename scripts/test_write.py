import os

output_dir = 'c:/Users/aminh/OneDrive/Desktop/Projects/HSE_Vision/outputs/visualizations'
os.makedirs(output_dir, exist_ok=True)

file_path = os.path.join(output_dir, 'test.txt')

print(f"Attempting to write to: {os.path.abspath(file_path)}")

try:
    with open(file_path, 'w') as f:
        f.write('This is a test.')
    print("File write operation completed.")
except Exception as e:
    print(f"An error occurred: {e}")

if os.path.exists(file_path):
    print("Verification successful: File exists.")
else:
    print("Verification failed: File does not exist.")