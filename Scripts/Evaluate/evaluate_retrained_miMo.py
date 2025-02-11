# compute the costs (deepcog cost and mae cost) for miMo and original deepcog and save them in a csv files
import multiprocessing
from multiprocessing import Pool
from Scripts.Evaluate.utils_deepcog import *

# Set multiprocessing start method
multiprocessing.set_start_method('spawn', force=True)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # Set the GPU card to use
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
model_deepcog_miMo_retrained = '../../../../oracle-data/serly/Scalable_dnn/Trained_models/DeepCOG_miMo_retrained/'
main_dir = f'../../../../oracle-data/serly/Scalable_dnn/MAE_plots/DeepCOG_miMo_vs_MiMo/'

# Function to process a single parameter combination
def process_combination(params):
    k, nr2, city = params
    cells_dir = f'../../../../oracle-data/serly/Scalable_dnn/LRP_scores/mimo_model/{city}/K_{k}/'
    cells = get_rows_Milan(5060, nr)
    length = 1780
    cluster_counts = np.load(os.path.join(cluster_directory, f'clusters_{city}_{k}.npy'))
    clustered_cells = {i: [] for i in range(k)}

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
            print(f"Number of cells for cluster {cluster_label} in K:{k} is set to {local_nr2} (nearest square number).")

        selected_cells = np.load(os.path.join(cells_dir, f'sorted_cells_{cluster_label}.npy')).tolist()[:local_num_cells]
    
        # Prepare data
        X_test, y_test, y_scalers = preprocess_cluster_data(selected_cells, cluster_cells, output_directory, city, lookback, local_nr2, data_type="test")
        model_miMo_path = os.path.join(model_deepcog_miMo_retrained, f'{city}/k_{k}/cluster_{cluster_label}_size_{local_num_cells}.h5')
        model_miMo = load_model_deepcog(model_miMo_path)
        predicted_miMo = model_miMo.predict(X_test)


        # After making predictions with the models
        predicted_miMo_unnorm = np.zeros_like(predicted_miMo)  # Initialize with same shape as predictions
        y_test_unnorm = np.zeros_like(y_test)

        # Unnormalize `predicted_miMo` using the scalers for each of the selected cells
        for i in range(predicted_miMo.shape[1]):
            cell_id = cluster_cells[i]  # Use the correct cell ID to get the scaler
            scaler = y_scalers[cell_id]
            # Apply inverse_transform to the entire time series for cell i
            predicted_miMo_unnorm[:, i] = scaler.inverse_transform(predicted_miMo[:, i].reshape(-1, 1)).flatten()
            y_test_unnorm[:, i] = scaler.inverse_transform(y_test[:, i].reshape(-1, 1)).flatten()

        X_test, y_test = X_test[:length], y_test[:length]
        predicted_miMo = predicted_miMo[:length]
        predicted_miMo_unnorm = predicted_miMo_unnorm[:length]
        y_test_unnorm = y_test_unnorm[:length]

        # Calculate costs
        costs_miMo = []
        for i in range(predicted_miMo.shape[1]):
            costs_miMo.append(compute_mae(predicted_miMo_unnorm[:, i], y_test_unnorm[:, i]))


        # Create a list of dictionaries
        data = [{'costs_miMo': miMo} for miMo in costs_miMo]

        # Define the CSV file name
        csv_path = os.path.join(main_dir, f'cvs/retrained_mimo/{city}/')
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        csv_file = os.path.join(csv_path,f'costs_comparison_K_{k}_{cluster_label}_size_{local_nr2}.csv')

        # Write the data to a CSV file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['costs_miMo'])
            writer.writeheader()
            writer.writerows(data)

        # Aggregate results
        avg_cost_miMo_total = np.mean(costs_miMo)


        results.append({
            'city': city, 'K': k, 'nr2': local_nr2, 
            'cluster': cluster_label,
            'avg_cost_miMo': [avg_cost_miMo_total]
        })

    return results

# Prepare combinations for multiprocessing
Ks = [2, 3, 4, 5, 6, 10, 15, 20]
nr2_values = [2, 3, 4, 5, 6, 7]
cities = ['Milan']
random_flags = [False]

combinations = [(k, nr2, city) for k in Ks for nr2 in nr2_values for city in cities]

# Run multiprocessing
if __name__ == '__main__':
    with Pool(100) as p:
        all_results = p.map(process_combination, combinations)
