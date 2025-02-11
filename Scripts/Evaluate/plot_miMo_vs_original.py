# compute the costs (deepcog cost and mae cost) for miMo and original deepcog and save them in a csv files
import multiprocessing
from multiprocessing import Pool
from Scripts.Evaluate.utils_deepcog import *

# Set multiprocessing start method
multiprocessing.set_start_method('spawn', force=True)
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # Set the GPU card to use
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
model_deepcog_miMo = '../../../../oracle-data/serly/Scalable_dnn/Trained_models/DeepCOG_miMo/'
model_deepcog_original = '../../../../oracle-data/serly/Scalable_dnn/Trained_models/DeepCOG_original/'
main_dir = f'../../../../oracle-data/serly/Scalable_dnn/MAE_plots/DeepCOG_miMo_vs_MiMo/'

# Function to process a single parameter combination
def process_combination(params):
    K, nr2, city, random_flag = params
    
    cells = get_rows_Paris(2700, nr) if city == 'Paris' else get_rows_Milan(5060, nr)
    length = 1780 if city == 'Milan' else 400
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
        X_test, y_test, y_scalers = preprocess_cluster_data(selected_cells, cluster_cells, output_directory, city, lookback, local_nr2, data_type="test")
        X_test_original, y_test_original, y_scalers_original = preprocess_cluster_data(cells, cells, output_directory, city, lookback, nr, data_type="test")

        model_miMo_path = os.path.join(model_deepcog_miMo, f'random_selection_{random_flag}/{city}/k_{K}/cluster_{cluster_label}_size_{local_num_cells}.h5')
        model_original_path = os.path.join(model_deepcog_original, f'{city}/deepcog_size_{nr}.h5')
        model_miMo = load_model_deepcog(model_miMo_path)
        predicted_miMo = model_miMo.predict(X_test)

        model_original = load_model_deepcog(model_original_path)
        predicted_original = model_original.predict(X_test_original)

        # After making predictions with the models
        predicted_miMo_unnorm = np.zeros_like(predicted_miMo)  # Initialize with same shape as predictions
        predicted_original_unnorm = np.zeros_like(predicted_original)
        y_test_unnorm = np.zeros_like(y_test)
        y_test_original_unnorm = np.zeros_like(y_test_original)

        # Unnormalize `predicted_miMo` using the scalers for each of the selected cells
        for i in range(predicted_miMo.shape[1]):
            cell_id = cluster_cells[i]  # Use the correct cell ID to get the scaler
            scaler = y_scalers[cell_id]
            # Apply inverse_transform to the entire time series for cell i
            predicted_miMo_unnorm[:, i] = scaler.inverse_transform(predicted_miMo[:, i].reshape(-1, 1)).flatten()
            y_test_unnorm[:, i] = scaler.inverse_transform(y_test[:, i].reshape(-1, 1)).flatten()

        # Unnormalize `predicted_original` using the scalers for each of the cells in the original dataset
        for i in range(predicted_original.shape[1]):
            cell_id = cells[i]  # Use the correct cell ID to get the scaler
            scaler = y_scalers_original[cell_id]
            # Apply inverse_transform to the entire time series for cell i
            predicted_original_unnorm[:, i] = scaler.inverse_transform(predicted_original[:, i].reshape(-1, 1)).flatten()
            y_test_original_unnorm[:, i] = scaler.inverse_transform(y_test_original[:, i].reshape(-1, 1)).flatten()

        X_test, y_test = X_test[:length], y_test[:length]
        X_test_original, y_test_original = X_test_original[:length], y_test_original[:length]
        predicted_miMo, predicted_original = predicted_miMo[:length], predicted_original[:length]
        predicted_miMo_unnorm, predicted_original_unnorm = predicted_miMo_unnorm[:length], predicted_original_unnorm[:length]
        y_test_unnorm, y_test_original_unnorm = y_test_unnorm[:length], y_test_original_unnorm[:length]

        cluster_indexes = np.where(np.isin(cells, cluster_cells))[0]
        # predicted_ts_original = predicted_original[:, cluster_indexes]
        predicted_ts_original_unnorm = predicted_original_unnorm[:, cluster_indexes]
        y_test_original_unnorm = y_test_original_unnorm[:, cluster_indexes]

        # Calculate costs
        costs_miMo, costs_original = [], []
        for i in range(predicted_miMo.shape[1]):
            costs_miMo.append(compute_mae(predicted_miMo_unnorm[:, i], y_test_unnorm[:, i]))
            costs_original.append(compute_mae(predicted_ts_original_unnorm[:, i], y_test_original_unnorm[:, i]))

        # Ensure both lists have the same length
        assert len(costs_miMo) == len(costs_original), "The lengths of costs_miMo and costs_original must be the same."

        # Create a list of dictionaries
        data = [{'costs_miMo': miMo, 'costs_original': original} for miMo, original in zip(costs_miMo, costs_original)]

        # Define the CSV file name
        csv_path = os.path.join(main_dir, f'cvs/{city}/{random_flag}/')
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        csv_file = os.path.join(csv_path,f'costs_comparison_K_{K}_{cluster_label}_size_{local_nr2}.csv')

        # Write the data to a CSV file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['costs_miMo', 'costs_original'])
            writer.writeheader()
            writer.writerows(data)

        # Aggregate results
        avg_cost_miMo_total = np.mean(costs_miMo)
        
        avg_cost_original_total = np.mean(costs_original)


        results.append({
            'city': city, 'K': K, 'nr2': local_nr2, 'random_flag': random_flag, 
            'cluster': cluster_label,
            'avg_cost_miMo': [avg_cost_miMo_total], 
            'avg_cost_original': [avg_cost_original_total]
        })

    return results

# Prepare combinations for multiprocessing
# Ks = [2, 3, 4, 5, 6, 10, 15, 20]
# nr2_values = [2, 3, 4, 5, 6, 7]
# cities = ['Paris', 'Milan']
# random_flags = [False, True]

Ks = [2, 3, 4, 5]
nr2_values = [2, 3, 4, 5, 6, 7]
cities = ['Paris']
random_flags = [True]
combinations = [(K, nr2, city, random_flag) for K in Ks for nr2 in nr2_values for city in cities for random_flag in random_flags]

# Run multiprocessing
if __name__ == '__main__':
    with Pool(100) as p:
        all_results = p.map(process_combination, combinations)
