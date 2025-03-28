# after running LRP on the initial miMo model, we find the best cells to train on. We then train a new model on these cells.

from utils_deepcog import *
import multiprocessing
from multiprocessing import Pool
from datetime import datetime
import math
import random 

# Set multiprocessing start method
multiprocessing.set_start_method('spawn', force=True)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # Set the GPU card to use
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Or '3' for FATAL logs only
np.set_printoptions(suppress=True)
physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)


def myMultiOpt(pair):

    # Record the start time
    city, k, nr2 = pair

    # Load cells based on the city
    cells = get_rows_Milan(5060, nr)


    model_dir = f'../../../../oracle-data/serly/Scalable_dnn/Trained_models/DeepCOG_miMo_retrained/{city}/k_{k}'
    cells_dir = f'../../../../oracle-data/serly/Scalable_dnn/LRP_scores/mimo_model/{city}/K_{k}/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Load cluster data
    cluster_counts = np.load(cluster_directory + f'clusters_{city}_{k}.npy')

    # Initialize a dictionary to store the cell IDs for each cluster
    clustered_cells = {i: [] for i in range(k)}

    # Categorize the cells into clusters
    for idx, cluster_label in enumerate(cluster_counts):
        cell_id = cells[idx]
        clustered_cells[cluster_label].append(cell_id)

    # Initialize a dictionary to store the models for each cluster
    cluster_models = {}

    # Loop through each cluster and train models with the specified percentage of cells
    for cluster_label in clustered_cells:
        # Define local copies of global variables
        local_nr2 = nr2
        local_num_cells = nr2 * nr2
        # Get the cell IDs for the current cluster
        cluster_cells = clustered_cells.get(cluster_label, [])
        if not cluster_cells:
            print(f"No cells found for Cluster {cluster_label}")
            continue

        total_cells = len(cluster_cells)  # Total number of cells in this cluster
        print(f"Cluster {cluster_label} has {total_cells} cells")

        # Ensure num_cells is a square number and <= total_cells
        if local_num_cells > total_cells:
            print(f"{cluster_label} has less BS than {local_num_cells} BSs.")
            # Find the largest perfect square less than or equal to total_cells
            largest_square = int(math.floor(math.sqrt(total_cells))) ** 2
            local_num_cells = largest_square
            local_nr2 = int(math.sqrt(local_num_cells))
            print(f"Number of cells for cluster {cluster_label} is set to {local_num_cells} (nearest square number).")

        # Train one model for each cluster with the input of shape (num_cells, lookback, 1) and output of shape (total_cells, 1)
        start_time = time.time()

        # select cells from sorted LRP scores
        selected_cells = np.load(os.path.join(cells_dir, f'sorted_cells_{cluster_label}.npy')).tolist()
        selected_cells = selected_cells[:local_num_cells]
        print(f"Selected number of cells for cluster {cluster_label}: {len(selected_cells)}")

        # Preprocess the training data for the selected cells
        X_train, y_train, _ = preprocess_cluster_data(selected_cells, cluster_cells, output_directory, city, lookback, local_nr2, data_type="train")
        
        model = model_deepcog(local_nr2, lookback, total_cells, neurons, ker_sz)

        # Train the model on the entire training dataset
        model.fit(X_train, y_train, epochs=epochs, batch_size=batchsize, verbose=0)

        # Store the model for this cluster and number of cells
        cluster_models[(cluster_label, local_num_cells)] = model

        # Save the model to disk
        model_filename = f"cluster_{cluster_label}_size_{local_num_cells}.h5"
        # dont save if the model exists
        if not os.path.exists(os.path.join(model_dir, model_filename)):
            model.save(os.path.join(model_dir, model_filename))

        end_time = time.time()
        training_duration = end_time - start_time
        print(f"Model for Cluster {cluster_label} with {local_num_cells} input size saved.")
        # Check if the file already exists
        log_file_path = os.path.join(model_dir, f'time_log_{cluster_label}_size_{local_num_cells}.txt')
        if not os.path.exists(log_file_path):
            # Open the file for writing only if it doesn't exist
            with open(log_file_path, 'a') as log_file:
                log_file.write(f'Training took {training_duration:.2f} seconds for Cluster {cluster_label} with size {local_num_cells}.\n')

# Global parameters
epochs = 20
batchsize = 32
neurons = 32
ker_sz = 3
multiple = False
lookback = 3
nr = 21
main_directory = '../../../../oracle-data/serly/Scalable_dnn/'
output_directory = '../../../../oracle-data/serly/Scalable_dnn/PerBS/'
cluster_directory = '../../../../oracle-data/serly/Scalable_dnn/cluster/'

if not os.path.exists(cluster_directory):
    os.makedirs(cluster_directory)

# Create list of parameter pairs for multiprocessing
pair_list = []
cities = ['Milan']
clusters = [2, 3, 4, 5, 6, 10, 15, 20]
# clusters = [15, 20]
gird_size = [2, 3, 4, 5, 6, 7]


for city in cities:
    for k in clusters:
        for nr2 in gird_size:
            pair_list.append((city, k, nr2))

if __name__ == '__main__':
    with Pool(30) as p:
        p.map(myMultiOpt, pair_list)
