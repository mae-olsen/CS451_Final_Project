import os

for root, dirs, files in os.walk(os.path.join('dataset', 'expression_labels')):
    for label_file in files:
        file_path = os.path.join(root, label_file)

        new_lines = []

        # Read and modify
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) == 5:  # valid YOLO line
                    parts[0] = '0'  # force class to 0
                    new_lines.append(" ".join(parts))

        name, ext = os.path.splitext(label_file)
        new_file_name = f"{name}_masked.txt"
        new_file_path = os.path.join(root, new_file_name)

        with open(new_file_path, "w") as f:
            f.write("\n".join(new_lines)+"\n")