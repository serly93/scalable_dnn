# evaluate lstm and save them in a csv files
import multiprocessing
from multiprocessing import Pool
from Scripts.Evaluate.utils_deepcog import *

# Set multiprocessing start method
multiprocessing.set_start_method('spawn', force=True)
os.environ['CUDA_VISIBLE_DEVICES'] = "3"  # Set the GPU card to use
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Or '3' for FATAL logs only
np.set_printoptions(suppress=True)
physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

# Set global parameters
lookback = 3
nr = 21

# Directories
output_directory = '../../../../oracle-data/serly/Scalable_dnn/PerBS/'
cluster_directory = '../../../../oracle-data/serly/Scalable_dnn/cluster/'
model_lstm_dir = '../../../../oracle-data/serly/Scalable_dnn/Trained_models/LSTM/'
main_dir = f'../../../../oracle-data/serly/Scalable_dnn/MAE_plots/DeepCOG_miMo_vs_MiMo/'

# Function to process a single parameter combination
def process_combination(params):
    K, nr2, city, random_flag = params
    
    cells = get_rows_Milan(5060, nr)
    length = 1780
    cluster_counts = np.load(os.path.join(cluster_directory, f'clusters_{city}_{K}.npy'))
    clustered_cells = {i: [] for i in range(K)}

    # Populate the cluster cells dictionary
    for idx, cluster_label in enumerate(cluster_counts):
        cell_id = cells[idx]
        clustered_cells[cluster_label].append(cell_id)

    results = []
    for cluster_label, cluster_cells in clustered_cells.items():
        local_nr2 = nr2
        local_num_cells = nr2 * nr2
        total_cells = len(cluster_cells)

        # Adjust if the number of cells exceeds available cells
        if local_num_cells > total_cells:
            largest_square = int(math.floor(math.sqrt(total_cells))) ** 2
            local_num_cells = largest_square
            local_nr2 = int(math.sqrt(local_num_cells))
            print(f"Number of cells for cluster {cluster_label} in K:{K} is set to {local_nr2} (nearest square number).")
        random.seed(42)
        selected_cells = (random.sample(cluster_cells, local_num_cells) if random_flag else 
                          np.load(os.path.join(cluster_directory, f'closest_to_centroid/closest_bs_{city}_{K}_{cluster_label}_100.npy')).tolist()[:local_num_cells])

        # Prepare data
        X_test, y_test, y_scalers = preprocess_cluster_data_lstm(selected_cells, cluster_cells, output_directory, city, lookback, data_type="test")

        model_lstm_path = os.path.join(model_lstm_dir, f'random_selection_{random_flag}/{city}/k_{K}/cluster_{cluster_label}_size_{local_num_cells}.h5')
        model_siMo = load_model_deepcog(model_lstm_path)
        predicted_siMo = model_siMo.predict(X_test)


        # After making predictions with the models
        predicted_siMo_unnorm = np.zeros_like(predicted_siMo)  # Initialize with same shape as predictions
        y_test_unnorm = np.zeros_like(y_test)

        # Unnormalize `predicted_miMo` using the scalers for each of the selected cells
        for i in range(predicted_siMo.shape[1]):
            cell_id = cluster_cells[i]  # Use the correct cell ID to get the scaler
            scaler = y_scalers[cell_id]
            # Apply inverse_transform to the entire time series for cell i
            predicted_siMo_unnorm[:, i] = scaler.inverse_transform(predicted_siMo[:, i].reshape(-1, 1)).flatten()
            y_test_unnorm[:, i] = scaler.inverse_transform(y_test[:, i].reshape(-1, 1)).flatten()


        X_test, y_test = X_test[:length], y_test[:length]
        predicted_siMo = predicted_siMo[:length]
        predicted_siMo_unnorm = predicted_siMo_unnorm[:length]
        y_test_unnorm = y_test_unnorm[:length]
        # Calculate costs
        costs_siMo = []
        for i in range(predicted_siMo.shape[1]):
            costs_siMo.append(compute_mae_slanum(predicted_siMo_unnorm[:, i], y_test_unnorm[:, i]))

        # Create a list of dictionaries
        data = [{'costs_siMo': siMo} for siMo in costs_siMo]

        # Define the CSV file name
        csv_path = os.path.join(main_dir, f'cvs/LSTM/{city}/{random_flag}/')
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        csv_file = os.path.join(csv_path,f'costs_lstm_K_{K}_{cluster_label}_size_{local_nr2}.csv')

        # Write the data to a CSV file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['costs_siMo'])
            writer.writeheader()
            writer.writerows(data)

        # Aggregate results
        avg_cost_siMo_total = np.mean([c[0] for c in costs_siMo])


        results.append({
            'city': city, 'K': K, 'nr2': local_nr2, 'random_flag': random_flag, 
            'cluster': cluster_label,
            'avg_cost_miMo': [avg_cost_siMo_total], 
        })

    return results

# Prepare combinations for multiprocessing
Ks = [2, 3, 4, 5, 6, 10, 15, 20]
nr2_values = [2, 3, 4, 5, 6, 7]
cities = ['Milan']
random_flags = [False]
combinations = [(K, nr2, city, random_flag) for K in Ks for nr2 in nr2_values for city in cities for random_flag in random_flags]

# Run multiprocessing
if __name__ == '__main__':
    with Pool(120) as p:
        all_results = p.map(process_combination, combinations)
