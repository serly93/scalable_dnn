# compute lrp for each cluster with all the BSs as inputs and all as outputs
import multiprocessing
from multiprocessing import Pool
from functions.utils_lrp import *


# Set multiprocessing start method
multiprocessing.set_start_method('spawn', force=True)
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  # Set the GPU card to use
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Or '3' for FATAL logs only
np.set_printoptions(suppress=True)
physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

# Set global parameters
lookback = 3
nr = 21

# Directories
main_directory = '../../../../oracle-data/serly/Scalable_dnn/'
dataset_directory = '../../../../oracle-data/serly/Scalable_dnn/PerBS/'
cluster_directory = '../../../../oracle-data/serly/Scalable_dnn/cluster/'
model_deepcog_mimo = '../../../../oracle-data/serly/Scalable_dnn/Trained_models/DeepCOG_MiMo_clusters/'
output_directory = '../../../../oracle-data/serly/Scalable_dnn/MAE_plots/DeepCOG_miMo_vs_MiMo/'

# Function to process a single parameter combination
def process_combination(params):
    k, city = params
    
    cells = get_rows_Milan(5060, nr)
    length = 1780
    cluster_counts = np.load(os.path.join(cluster_directory, f'clusters_{city}_{k}.npy'))
    clustered_cells = {i: [] for i in range(k)}

    # Populate the cluster cells dictionary
    for idx, cluster_label in enumerate(cluster_counts):
        cell_id = cells[idx]
        clustered_cells[cluster_label].append(cell_id)

    for cluster_label, cluster_cells in clustered_cells.items():

        total_cells = len(cluster_cells)

        # if total_cells is not a square number, find the nearest square number
        local_num_cells = int(math.sqrt(total_cells)) ** 2
        for round in range(1, 4):
            local_cells = np.load(os.path.join(model_deepcog_mimo, f'{city}/k_{k}/local_cells_{cluster_label}_size_all_{total_cells}_round_{round}.npy'))

            print(f"Number of cells for cluster {cluster_label} is set to {local_num_cells} (nearest square number).")
            square_root = int(math.sqrt(local_num_cells))
            # Train one model for each cluster with the input of shape (num_cells, lookback, 1) and output of shape (total_cells, 1)

            # Prepare data
            X_test, y_test, y_scalers = preprocess_cluster_data(local_cells, cluster_cells, dataset_directory, city, lookback, square_root, data_type="test")

            model_mimo_path = os.path.join(model_deepcog_mimo, f'{city}/k_{k}/cluster_{cluster_label}_size_all_{total_cells}_round_{round}.h5')
            model_mimo = load_model_deepcog(model_mimo_path)
            predicted_mimo = model_mimo(X_test)


            # Initialize relevance at the output layer
            relevance_output = np.zeros_like(predicted_mimo.numpy())
            relevance_output[:, :] = predicted_mimo[:, :]

            # Compute LRP
            lrp_map = compute_lrp(model_mimo, X_test, relevance_output)

            # Save the lrp scores
            lrp_dir = os.path.join(main_directory, f'LRP_scores/mimo_model/{city}/K_{k}')
            if not os.path.exists(lrp_dir):
                os.makedirs(lrp_dir)
            np.save(os.path.join(lrp_dir, f'cluster_{cluster_label}_size_{total_cells}_round_{round}.npy'), lrp_map)

            # Visualize relevance for the first sample
            # Aggregate relevance over the temporal dimension (lookback)
            # aggregated_lrp_map = np.sum(lrp_map.numpy(), axis=1)
            # print("Aggregated LRP Map Shape:", aggregated_lrp_map.shape)  # Should match input shape (10, 2, 2, 1)
            # plt.imshow(aggregated_lrp_map[0, :, :, 0], cmap='hot', interpolation='nearest')
            # plt.colorbar()
            # plt.title(f"Relevance Map for Cluster {cluster_label} in {city} with {total_cells} cells")
            # plt.savefig(os.path.join(lrp_dir, f'cluster_{cluster_label}_size_{total_cells}.png'))
            # plt.close()

            # # Print results
            # print("LRP Map Shape:", lrp_map.shape)  # Should match input shape (batch, lookback, size, size, 1)


# Prepare combinations for multiprocessing
# Ks = [2, 3, 4, 5, 6, 10, 15, 20]
Ks = [2, 3, 4, 5]
cities = ['Milan']
combinations = [(k, city) for k in Ks for city in cities]

# Run multiprocessing
if __name__ == '__main__':
    with Pool(100) as p:
        all_results = p.map(process_combination, combinations)
