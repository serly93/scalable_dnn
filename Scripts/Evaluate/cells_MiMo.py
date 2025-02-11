# save the selected cells for the mimo model
# Import required libraries
from Scripts.Evaluate.utils_deepcog import *
import multiprocessing
from multiprocessing import Pool
from datetime import datetime
import math
import random  # Make sure random is imported

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
    city, k = pair

    # Load cells based on the city
    cells = get_rows_Milan(5060, nr)

    output_directory = f'../../../../oracle-data/serly/Scalable_dnn/Trained_models/DeepCOG_MiMo_clusters/{city}/k_{k}/'
    # Load cluster data
    cluster_counts = np.load(cluster_directory + f'clusters_{city}_{k}.npy')

    # Initialize a dictionary to store the cell IDs for each cluster
    clustered_cells = {i: [] for i in range(k)}

    # Categorize the cells into clusters
    for idx, cluster_label in enumerate(cluster_counts):
        cell_id = cells[idx]
        clustered_cells[cluster_label].append(cell_id)


    # Loop through each cluster and train models with the specified percentage of cells
    for cluster_label in clustered_cells:

        # Get the cell IDs for the current cluster
        cluster_cells = clustered_cells.get(cluster_label, [])
        if not cluster_cells:
            print(f"No cells found for Cluster {cluster_label}")
            continue

        total_cells = len(cluster_cells)  # Total number of cells in this cluster
        # if total_cells is not a square number, find the nearest square number
        local_num_cells = int(math.sqrt(total_cells)) ** 2
        local_cells = cluster_cells[:local_num_cells]
        # save the cells
        np.save(output_directory + f'cluster_cells_{cluster_label}.npy', local_cells)

# Global parameters
epochs = 20
batchsize = 32
neurons = 32
ker_sz = 3
multiple = False
lookback = 3
nr = 21
main_directory = '../../../../oracle-data/serly/Scalable_dnn/'
cluster_directory = '../../../../oracle-data/serly/Scalable_dnn/cluster/'

if not os.path.exists(cluster_directory):
    os.makedirs(cluster_directory)

# Create list of parameter pairs for multiprocessing
pair_list = []
cities = ['Milan']
clusters = [2, 3, 4, 5, 6, 10, 15, 20]

for city in cities:
    for k in clusters:
        pair_list.append((city, k))

if __name__ == '__main__':
    with Pool(30) as p:
        p.map(myMultiOpt, pair_list)
