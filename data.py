import os

dataset_dir = r"C:\Users\astef\OneDrive\Desktop\Projects\dotCon\A4_dataset_ALL"

folders = sorted(os.listdir(dataset_dir))

print(f"{'Folder':<12} {'Image Count'}")
print("-" * 25)

total = 0
for folder in folders:
    folder_path = os.path.join(dataset_dir, folder)
    if os.path.isdir(folder_path):
        count = len([f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"{folder:<12} {count}")
        total += count

print("-" * 25)
print(f"{'TOTAL':<12} {total}")