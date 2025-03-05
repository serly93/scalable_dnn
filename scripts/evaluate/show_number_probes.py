# Import required libraries
import os
import pandas as pd
import numpy as np
import csv
import math
import random
import multiprocessing
from multiprocessing import Pool
from datetime import datetime
import tensorflow as tf
from Scripts.Evaluate.utils_deepcog import *

# Define the main directories
cluster_directory = '../../../../oracle-data/serly/Scalable_dnn/cluster/'
main_directory = '../../../../oracle-data/serly/Scalable_dnn/'
output_csv_path = os.path.join(main_directory, 'MAE_plots/DeepCOG_miMo_vs_MiMo/cvs/')

# Function for multiprocessing optimization
def myMultiOpt(pair, summary_list):
    city, k, nr2, random_flag = pair

    # Load cells based on the city
    cells = get_rows_Milan(5060, nr)

    # Load cluster data
    cluster_counts = np.load(cluster_directory + f'clusters_{city}_{k}.npy')

    # Initialize a dictionary to store the cell IDs for each cluster
    clustered_cells = {i: [] for i in range(k)}

    # Categorize the cells into clusters
    for idx, cluster_label in enumerate(cluster_counts):
        cell_id = cells[idx]
        clustered_cells[cluster_label].append(cell_id)

    # Initialize local sum for the current configuration
    local_sum_cells = 0

    # Loop through each cluster and train models with the specified percentage of cells
    for cluster_label in clustered_cells:
        local_nr2 = nr2
        local_num_cells = nr2 * nr2
        cluster_cells = clustered_cells.get(cluster_label, [])
        if not cluster_cells:
            print(f"No cells found for Cluster {cluster_label}")
            continue

        total_cells = len(cluster_cells)

        # Ensure num_cells is a square number and <= total_cells
        if local_num_cells > total_cells:
            largest_square = int(math.floor(math.sqrt(total_cells))) ** 2
            local_num_cells = largest_square
            local_nr2 = int(math.sqrt(local_num_cells))

        # Accumulate local_num_cells to the local sum
        local_sum_cells += local_num_cells

    # Append the current configuration to the summary list
    summary_list.append((city, k, nr2, local_sum_cells))

# Global parameters
multiple = False
lookback = 3
nr = 21

# Create list of parameter pairs for multiprocessing
pair_list = []
cities = ['Milan']
clusters = [2, 3, 4, 5, 6, 10, 15, 20]
grid_size = [2, 3, 4, 5, 6, 7]
random_flags = [False]

for city in cities:
    for k in clusters:
        for nr2 in grid_size:
            for random_flag in random_flags:
                pair_list.append((city, k, nr2, random_flag))

if __name__ == '__main__':
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    
    # Manager for the global list for summary
    manager = multiprocessing.Manager()
    summary_list = manager.list()
    
    # Use Pool for parallel execution
    with Pool(30) as p:
        p.starmap(myMultiOpt, [(pair, summary_list) for pair in pair_list])
    
    # Convert the summary list to a DataFrame
    summary_df = pd.DataFrame(list(summary_list), columns=['City', 'K', 'NR2', 'Actual Num Cells'])
    
    # Save the summary to CSV file
    csv_file_name = f'local_num_cells_summary_{city}.csv'
    summary_df.to_csv(os.path.join(output_csv_path, csv_file_name), index=False)

    # Print completion message
    print(f"\nSummary saved to CSV file: {output_csv_path}")

