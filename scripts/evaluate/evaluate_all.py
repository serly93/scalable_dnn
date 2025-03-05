import os
import time
import csv
import math
import random
import numpy as np
import multiprocessing
from multiprocessing import Pool
from sklearn.metrics import mean_absolute_error  # No longer used, but kept if needed later.
import tensorflow as tf

# Import your helper functions from your utility modules.
from utils_deepcog import *

# ----------------------------
# Setup GPU and Environment
# ----------------------------
multiprocessing.set_start_method('spawn', force=True)
os.environ['CUDA_VISIBLE_DEVICES'] = "3"  # adjust as needed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(suppress=True)
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

# ----------------------------
# Global Parameters & Directories
# ----------------------------
city = "Milan"        
lookback = 3           # lookback window for time series
nr = 21                # used to generate cell lists
test_length = 1780

# Directories for data and clusters
output_directory = "../../../../oracle-data/serly/Scalable_dnn/PerBS/"
cluster_directory = "../../../../oracle-data/serly/Scalable_dnn/cluster/"

# For saving evaluation logs per strategy.
eval_logs_dir = "../../../../oracle-data/serly/Scalable_dnn/eval_logs/"

# ----------------------------
# Function: Evaluate All Strategies for One Combination
# ----------------------------
def process_combination(params):
    # params: (K, nr2, city, random_flag)
    KK, nr2, city, random_flag = params

    # Define model directories for each strategy.
    model_dirs = {
        "LRP-Cluster-DNN": f"../../../../oracle-data/serly/Scalable_dnn/Trained_models/DeepCOG_miMo_retrained/{city}/k_{KK}/",
        "Centroid-Cluster-DNN": f"../../../../oracle-data/serly/Scalable_dnn/Trained_models/DeepCOG_miMo/random_selection_False/{city}/k_{KK}/",
        "Cluster-LSTM": f"../../../../oracle-data/serly/Scalable_dnn/Trained_models/LSTM/random_selection_False/{city}/k_{KK}/",
        "LSTM-PerBS": f"../../../../oracle-data/serly/Scalable_dnn/Trained_models/Per_BS/LSTM/{city}/",
        "Global-DNN": f"../../../../oracle-data/serly/Scalable_dnn/Trained_models/DeepCOG_original/{city}/"
    }

    # Get the complete list of cell IDs.
    cells = get_rows_Milan(5060, nr)
    length = 1780

    # Load cluster assignments.
    cluster_counts = np.load(os.path.join(cluster_directory, f"clusters_{city}_{KK}.npy"))
    clustered_cells = {i: [] for i in range(KK)}
    for idx, cl in enumerate(cluster_counts):
        clustered_cells[cl].append(cells[idx])
    
    # Prepare a results list to store evaluation logs for each strategy.
    all_logs = []

    # For each cluster (subcluster), evaluate each strategy.
    for cluster_label, cell_list in clustered_cells.items():
        # Determine the number of cells for a square subset.
        local_num_cells = nr2 * nr2
        total_cells = len(cell_list)
        if local_num_cells > total_cells:
            largest_square = int(math.floor(math.sqrt(total_cells))) ** 2
            local_num_cells = largest_square
            nr2 = int(math.sqrt(local_num_cells))
            print(f"Adjusted cluster {cluster_label} in K:{KK} to {nr2}x{nr2} (nearest square).")
        
        # 1. LRP-Cluster-DNN: Use sorted cells from LRP scores.
        try:
            cells_dir = f"../../../../oracle-data/serly/Scalable_dnn/LRP_scores/mimo_model/{city}/K_{KK}/"
            sorted_cells = np.load(os.path.join(cells_dir, f"sorted_cells_{cluster_label}.npy")).tolist()
        except Exception as e:
            print(f"LRP sorted cells not found for cluster {cluster_label}: {e}")
            sorted_cells = random.sample(cell_list, local_num_cells)
        selected_cells_lrp = sorted_cells[:local_num_cells]
        X_test, y_test, _ = preprocess_cluster_data(selected_cells_lrp, cell_list, output_directory, city, lookback, nr2, data_type="test")
        model_path = os.path.join(model_dirs["LRP-Cluster-DNN"], f"cluster_{cluster_label}_size_{local_num_cells}.h5")
        start_time = time.time()
        model = load_model_deepcog(model_path)
        _ = model.predict(X_test)
        inf_time = time.time() - start_time
        log_entry = {
            "strategy": "LRP-Cluster-DNN",
            "city": city,
            "K": KK,
            "cluster": cluster_label,
            "inference_time_sec": inf_time,
        }
        all_logs.append(log_entry)
        save_log(model_dirs["LRP-Cluster-DNN"], log_entry)

        # 2. Centroid-Cluster-DNN: Use cells selected based on centroid.
        selected_cells_centroid = (random.sample(cell_list, local_num_cells) if random_flag 
                                   else np.load(os.path.join(cluster_directory, f"closest_to_centroid/closest_bs_{city}_{KK}_{cluster_label}_100.npy")).tolist()[:local_num_cells])
        X_test, y_test, _ = preprocess_cluster_data(selected_cells_centroid, cell_list, output_directory, city, lookback, nr2, data_type="test")
        model_path = os.path.join(model_dirs["Centroid-Cluster-DNN"], f"cluster_{cluster_label}_size_{local_num_cells}.h5")
        start_time = time.time()
        model = load_model_deepcog(model_path)
        _ = model.predict(X_test)
        inf_time = time.time() - start_time
        log_entry = {
            "strategy": "Centroid-Cluster-DNN",
            "city": city,
            "K": KK,
            "cluster": cluster_label,
            "inference_time_sec": inf_time,
        }
        all_logs.append(log_entry)
        save_log(model_dirs["Centroid-Cluster-DNN"], log_entry)

        # 3. Cluster-LSTM: Use LSTM-based cluster model.
        X_test, y_test, _ = preprocess_cluster_data_lstm(selected_cells_centroid, cell_list, output_directory, city, lookback, data_type="test")
        model_path = os.path.join(model_dirs["Cluster-LSTM"], f"cluster_{cluster_label}_size_{local_num_cells}.h5")
        start_time = time.time()
        model = load_model_deepcog(model_path)
        _ = model.predict(X_test)
        inf_time = time.time() - start_time
        log_entry = {
            "strategy": "Cluster-LSTM",
            "city": city,
            "K": KK,
            "cluster": cluster_label,
            "inference_time_sec": inf_time,
        }
        all_logs.append(log_entry)
        save_log(model_dirs["Cluster-LSTM"], log_entry)

        # 4. LSTM-PerBS: Evaluate each BS in the subcluster individually and sum their inference times.
        perbs_time = []
        model_folder = model_dirs["LSTM-PerBS"]
        for cell in cell_list:
            try:
                test_X, test_Y, _ = preprocess_data(cell, output_directory, city, lookback, data_type="test")
            except Exception as e:
                print(f"SISO preprocessing failed for cell {cell}: {e}")
                continue
            test_X = test_X[:length]
            test_Y = test_Y[:length]
            model_path_cell = os.path.join(model_folder, f"model_{cell}.h5")
            if not os.path.exists(model_path_cell):
                continue
            start_cell = time.time()
            model = tf.keras.models.load_model(model_path_cell)
            _ = model.predict(test_X)
            cell_time = time.time() - start_cell
            perbs_time.append(cell_time)
        total_time = sum(perbs_time) if perbs_time else None
        log_entry = {
            "strategy": "LSTM-PerBS",
            "city": city,
            "K": KK,
            "cluster": cluster_label,
            "inference_time_sec": total_time,
        }
        all_logs.append(log_entry)
        save_log(model_folder, log_entry)

        # 5. Global-DNN: Evaluate the global model on all cells.
        X_test, y_test, _ = preprocess_cluster_data(cells, cells, output_directory, city, lookback, nr, data_type="test")
        model_path = os.path.join(model_dirs["Global-DNN"], f"deepcog_size_{nr}.h5")
        start_time = time.time()
        model = load_model_deepcog(model_path)
        _ = model.predict(X_test)
        inf_time = time.time() - start_time
        log_entry = {
            "strategy": "Global-DNN",
            "city": city,
            "K": KK,
            "cluster": "global",
            "inference_time_sec": inf_time,
        }
        all_logs.append(log_entry)
        save_log(model_dirs["Global-DNN"], log_entry)

    # Aggregate and log the summed inference times per strategy across all subclusters.
    strategy_totals = {}
    for log in all_logs:
        strat = log["strategy"]
        if log["inference_time_sec"] is not None:
            strategy_totals[strat] = strategy_totals.get(strat, 0) + log["inference_time_sec"]
    for strat, total in strategy_totals.items():
        summary_log = {
            "strategy": strat,
            "city": city,
            "K": KK,
            "cluster": "sum",
            "inference_time_sec": total,
        }
        all_logs.append(summary_log)
        # Save the summary log in the corresponding model directory (or eval_logs_dir if not found).
        log_dir = model_dirs[strat] if strat in model_dirs else eval_logs_dir
        save_log(log_dir, summary_log)

    return all_logs

# ----------------------------
# Helper Function: Save Log to a CSV File in a Given Directory
# ----------------------------
def save_log(model_dir, log_entry):
    # Ensure a logs folder exists inside the given model directory.
    log_folder = os.path.join(model_dir, "evaluation_logs")
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    # CSV header without mae, timestamp, or model location.
    filename = f"eval_{log_entry['strategy']}_cluster_{log_entry['cluster']}.csv"
    filepath = os.path.join(log_folder, filename)
    file_exists = os.path.exists(filepath)
    with open(filepath, "a", newline="") as csvfile:
        fieldnames = ["strategy", "city", "K", "cluster", "inference_time_sec"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

# ----------------------------
# Main Function: Run Evaluations via Multiprocessing
# ----------------------------
def main():
    # Prepare combinations: here we vary K, nr2, city, random_flag.
    Ks = [2]
    nr2_values = [6]
    cities = [city]  
    random_flags = [False]  # adjust if needed
    combinations = [(KK, nr2, c, rf) for KK in Ks for nr2 in nr2_values for c in cities for rf in random_flags]
    
    # Use a multiprocessing pool to process combinations.
    with Pool(10) as pool:
        all_results = pool.map(process_combination, combinations)
    
    # Flatten the results list.
    flattened = [entry for sublist in all_results for entry in sublist]
    # Save overall summary to a CSV.
    summary_csv = os.path.join(eval_logs_dir, "overall_evaluation_summary.csv")
    os.makedirs(eval_logs_dir, exist_ok=True)
    with open(summary_csv, "w", newline="") as f:
        fieldnames = ["strategy", "city", "K", "cluster", "inference_time_sec"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in flattened:
            writer.writerow(row)
    print(f"Overall evaluation summary saved to {summary_csv}")

if __name__ == "__main__":
    main()
