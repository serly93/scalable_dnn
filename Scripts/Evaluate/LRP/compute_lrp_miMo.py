# compute the costs (deepcog cost and mae cost) for miMo and original deepcog and save them in a csv files
import multiprocessing
from multiprocessing import Pool
from functions.utils_lrp import *


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
dataset_directory = '../../../../oracle-data/serly/Scalable_dnn/PerBS/'
cluster_directory = '../../../../oracle-data/serly/Scalable_dnn/cluster/'
model_deepcog_miMo = '../../../../oracle-data/serly/Scalable_dnn/Trained_models/DeepCOG_miMo/'
model_deepcog_original = '../../../../oracle-data/serly/Scalable_dnn/Trained_models/DeepCOG_original/'
output_directory = '../../../../oracle-data/serly/Scalable_dnn/MAE_plots/DeepCOG_miMo_vs_MiMo/'
main_directory = '../../../../oracle-data/serly/Scalable_dnn/'

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
        X_test, y_test, y_scalers = preprocess_cluster_data(selected_cells, cluster_cells, dataset_directory, city, lookback, local_nr2, data_type="test")
        # X_test_original, y_test_original, y_scalers_original = preprocess_cluster_data(cells, cells, dataset_directory, city, lookback, nr, data_type="test")

        model_miMo_path = os.path.join(model_deepcog_miMo, f'random_selection_{random_flag}/{city}/k_{K}/cluster_{cluster_label}_size_{local_num_cells}.h5')
        model_miMo = load_model_deepcog(model_miMo_path)
        predicted_miMo = model_miMo(X_test)

        # model_original_path = os.path.join(model_deepcog_original, f'{city}/deepcog_size_{nr}.h5')
        # model_original = load_model_deepcog(model_original_path)
        # predicted_original = model_original(X_test_original)


        # Initialize relevance at the output layer (focus on class 0)
        relevance_output = np.zeros_like(predicted_miMo.numpy())
        target_class = 0  # Class to explain
        relevance_output[:, :] = predicted_miMo[:, :]

        # Compute LRP
        lrp_map = compute_lrp(model_miMo, X_test, relevance_output)

        # Save the lrp scores
        lrp_dir = os.path.join(main_directory, f'LRP_scores/miMo_model/{city}/K_{K}')
        if not os.path.exists(lrp_dir):
            os.makedirs(lrp_dir)
        np.save(os.path.join(lrp_dir, f'cluster_{cluster_label}_size_{local_num_cells}.npy'), lrp_map)
        # Visualize relevance for the first sample
        import matplotlib.pyplot as plt

        # Aggregate relevance over the temporal dimension (lookback)
        aggregated_lrp_map = np.sum(lrp_map.numpy(), axis=1)
        print("Aggregated LRP Map Shape:", aggregated_lrp_map.shape)  # Should match input shape (10, 2, 2, 1)
        plt.imshow(aggregated_lrp_map[0, :, :, 0], cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title("Relevance Map for First Sample")
        plt.savefig(os.path.join(lrp_dir, f'cluster_{cluster_label}_size_{local_num_cells}.png'))
        plt.close()

        # Print results
        print("LRP Map Shape:", lrp_map.shape)  # Should match input shape (batch, lookback, size, size, 1)


# Prepare combinations for multiprocessing
Ks = [2, 3, 4, 5, 6, 10, 15, 20]
nr2_values = [2, 3, 4, 5, 6, 7]

cities = ['Milan']
random_flags = [False]
combinations = [(K, nr2, city, random_flag) for K in Ks for nr2 in nr2_values for city in cities for random_flag in random_flags]

# Run multiprocessing
if __name__ == '__main__':
    with Pool(100) as p:
        all_results = p.map(process_combination, combinations)
