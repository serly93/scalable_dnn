import os
import re
import glob
import numpy as np
from utils_deepcog import get_rows_Milan  # Import BS retrieval functions

# User input for city, cluster (k), and grid size
city = input("Enter city (Milan): ").strip()
if city not in ["EUMA", "Milan"]:
    print("Invalid city! Please enter either 'EUMA' or 'Milan'.")
    exit()

try:
    k = int(input("Enter total number of clusters (k): "))
    grid_size = int(input("Enter grid size: "))
except ValueError:
    print("Invalid input! k and grid size must be numbers.")
    exit()

# Define base directories for each model
models = {
    "LRP-Cluster-DNN": f"../../../../oracle-data/serly/Scalable_dnn/Trained_models/DeepCOG_miMo_retrained/{city}/k_{k}/",
    "Centroid-Cluster-DNN": f"../../../../oracle-data/serly/Scalable_dnn/Trained_models/DeepCOG_miMo/random_selection_False/{city}/k_{k}/",
    "Cluster-LSTM": f"../../../../oracle-data/serly/Scalable_dnn/Trained_models/LSTM/random_selection_False/{city}/k_{k}/",
    "LSTM-PerBS": f"../../../../oracle-data/serly/Scalable_dnn/Trained_models/Per_BS/LSTM/{city}/",
    "Global-DNN": f"../../../../oracle-data/serly/Scalable_dnn/Trained_models/DeepCOG_original/{city}/"
}

cluster_directory = "../../../../oracle-data/serly/Scalable_dnn/cluster/"

# Regular expressions for log files
cluster_log_pattern = re.compile(r'time_log_(\d+)_size_(\d+)\.txt')  # e.g., time_log_0_size_1.txt
global_log_pattern = re.compile(r'time_log_size_(\d+)\.txt')         # e.g., time_log_size_21.txt
perbs_log_pattern = re.compile(r"time_log_(\d+)\.txt")               # e.g., time_log_1234.txt (Per-BS)

# Dictionary to store total training times per model
training_times = {}

print(f"\n### Searching Logs for {city} | Total Clusters: {k} | Grid Size: {grid_size} ###\n")

# Load cluster file
cluster_assignment_file = os.path.join(cluster_directory, f'clusters_{city}_{k}.npy')
if not os.path.exists(cluster_assignment_file):
    print(f"Cluster file not found: {cluster_assignment_file}")
    exit()

cluster_counts = np.load(cluster_assignment_file)

# Get all BS IDs in the city
all_cells = get_rows_Milan(5060, 21)

# Dictionary to store BS IDs for each subcluster
clustered_cells = {subcluster_id: [] for subcluster_id in range(k)}

# Assign each BS to its subcluster
for idx, cluster_label in enumerate(cluster_counts):
    cell_id = all_cells[idx]
    clustered_cells[cluster_label].append(cell_id)

# Debug info
print(f"Subcluster Keys: {list(clustered_cells.keys())}")
print(f"Example cluster 0 cell IDs: {clustered_cells.get(0, [])[:5]} ...")

# Process all cluster-based models (except Per-BS) for logs
for model, base_dir in models.items():
    if model == "LSTM-PerBS":
        # We'll handle LSTM-PerBS separately below
        continue

    print(f"\n Searching logs in: {base_dir} for {model}")

    if not os.path.exists(base_dir):
        print(f"⚠ WARNING: Directory {base_dir} does not exist for {model}!")
        training_times[model] = 0
        continue

    training_times[model] = 0  # Initialize total time

    # Find logs in directory
    log_files = glob.glob(os.path.join(base_dir, "time_log_*.txt"))
    if not log_files:
        print(f"⚠ No log files found in {base_dir} for {model}")
        continue

    print(f"✔ Found {len(log_files)} log files in {model}")

    for log_file in log_files:
        filename = os.path.basename(log_file)
        match_cluster = cluster_log_pattern.search(filename)
        match_global = global_log_pattern.search(filename)

        if match_cluster:
            # e.g., time_log_0_size_3.txt
            cluster_label = int(match_cluster.group(1))   # Extract subcluster
            file_grid_size = int(match_cluster.group(2))    # Extract grid size

            if file_grid_size != grid_size:
                continue  # Skip logs that don't match the chosen grid size

            print(f"  Processing: Subcluster {cluster_label}, Grid {file_grid_size}")

        elif match_global and model == "Global-DNN":
            # For Global-DNN, always use log file with grid size 21 regardless of user input.
            file_grid_size = int(match_global.group(1))
            if file_grid_size != 21:
                continue
            print(f"  Processing Global Model, Grid {file_grid_size}")

        else:
            # Skip files that don't match any expected pattern
            continue

        # Read training time from the log file (only once)
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if "Training took" in line:
                        time_sec = float(line.split(" ")[2])  # Extract time in seconds
                        training_times[model] += time_sec
                        print(f"  Added {time_sec:.2f} sec to {model}")
                        break
        except Exception as e:
            print(f"⚠ Error reading {log_file}: {e}")

# ============== Extract Clustering Time ==============
# For cluster-based models (LRP-Cluster-DNN, Centroid-Cluster-DNN, Cluster-LSTM), 
# extract the clustering overhead from the cluster log file.
clustering_log_file = os.path.join(cluster_directory, f'log_{city}_{k}.txt')
clustering_time = None
if os.path.exists(clustering_log_file):
    try:
        with open(clustering_log_file, 'r') as f:
            for line in f:
                if "Time taken for clustering:" in line:
                    clustering_time = float(line.split("Time taken for clustering:")[1].strip().split(" ")[0])
                    print(f"\n🔍 Clustering time extracted: {clustering_time:.2f} sec from {clustering_log_file}")
                    break
        if clustering_time is None:
            print(f"⚠ Could not find clustering time in {clustering_log_file}")
    except Exception as e:
        print(f"⚠ Error reading clustering log {clustering_log_file}: {e}")
else:
    print(f"⚠ Clustering log file not found: {clustering_log_file}")

# ============== LSTM-PerBS subcluster sums ==============
# For each subcluster (0 to k-1), sum the logs of the cell IDs that belong to it
log_dir = models["LSTM-PerBS"]
print(f"\n🔍 Now calculating Per-BS times in: {log_dir}")

if not os.path.exists(log_dir):
    print(f"⚠ Directory not found for LSTM-PerBS: {log_dir}")
    training_times["LSTM-PerBS"] = 0
else:
    # We'll store subcluster sums in a dictionary
    perbs_subcluster_sums = {sub_id: 0 for sub_id in range(k)}

    # Retrieve all possible logs
    perbs_logs = glob.glob(os.path.join(log_dir, "time_log_*.txt"))
    if not perbs_logs:
        print(f"⚠ No Per-BS logs found in {log_dir}")
    else:
        print(f"✔ Found {len(perbs_logs)} Per-BS time logs")

        # For each subcluster, sum up times from matching BS logs
        for sub_id in range(k):
            subcluster_bs_ids = clustered_cells[sub_id]  # all cell IDs in subcluster sub_id
            subcluster_time = 0

            for log_file in perbs_logs:
                filename = os.path.basename(log_file)
                match = perbs_log_pattern.search(filename)  # e.g., time_log_1234.txt
                if match:
                    cell_id = int(match.group(1))

                    if cell_id in subcluster_bs_ids:
                        try:
                            with open(log_file, 'r') as f:
                                for line in f:
                                    if "Training took" in line:
                                        time_sec = float(line.split(" ")[2])
                                        subcluster_time += time_sec
                                        break
                        except Exception as e:
                            print(f"⚠ Error reading {log_file}: {e}")

            # Store the sum for this subcluster
            perbs_subcluster_sums[sub_id] = subcluster_time

    # Optionally: sum total across all subclusters
    total_perbs_time = sum(perbs_subcluster_sums.values())
    training_times["LSTM-PerBS"] = total_perbs_time

    print("\n### LSTM-PerBS Subcluster Results ###")
    for sub_id, val in perbs_subcluster_sums.items():
        print(f"  Subcluster {sub_id} → {val:.2f} sec")
    print(f"  Total (all subclusters) → {total_perbs_time:.2f} sec")

# ============== Final Output ==============
print(f"\n### Final Training Time Summary for {city} | Total Clusters: {k} | Grid Size: {grid_size} ###\n")
for model, total_time in training_times.items():
    if total_time == 0:
        print(f"⚠ No training time data for {model}")
        continue
    print(f"  {model} → Total Training Time: {total_time:.2f} sec")

# Print the clustering time separately for cluster-based models
if clustering_time is not None:
    print(f"\n🔍 Clustering Overhead for cluster-based models (LRP-Cluster-DNN, Centroid-Cluster-DNN, Cluster-LSTM): {clustering_time:.2f} sec")
    # Note:
    # The clustering time represents the duration required to group Base Stations into clusters.
    # It is a pre-processing or inference overhead. If clustering is performed at inference time,
    # then it should be considered as part of the inference time. However, if it's done offline once,
    # then it's more of a one-time pre-processing cost rather than a recurring inference expense.
else:
    print("\n⚠ No clustering time data available.")
