# clusters the timeseries of the nxn grid into K clusters using DTW
# saves the cluster labels, centroids, silhouette score, and plots and heatmaps of the clusters, cluster counts,
# and logs the silhouette score and time taken for clustering
from utils import *
import multiprocessing
from multiprocessing import Pool
import time
from datetime import datetime
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
from sklearn.metrics import silhouette_score  # Use scikit-learn's silhouette_score
import matplotlib.cm as cm

multiprocessing.set_start_method('spawn', force=True)
os.environ['CUDA_VISIBLE_DEVICES'] = "0" # set the gpu card to use
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '3' for FATAL logs only
np.set_printoptions(suppress=True)
physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)
nr = 21 # Global grid size

x1 = [0]
split_ratio = 0.8
num_cells = nr * nr

main_directory = '../../../../oracle-data/serly/TMC_data/'
output_directory = '../../../../oracle-data/serly/MoE_data/PerBS/'
cluster_directory = '../../../../oracle-data/serly/MoE_data/cluster/'
if not os.path.exists(cluster_directory):
    os.makedirs(cluster_directory)

def myMultiOpt(pair):
    bs_timeseries = []
    # Record the start time
    start_time = time.time()
    city, K = pair
  
    cells = get_rows_Milan(5060, nr)
    
    for cell in cells:
        ts = np.load(os.path.join(output_directory, f'{city}/train_{cell}.npy'))
        bs_timeseries.append(ts)

    # Convert to numpy array of shape (num_cells, time_points, 1)
    bs_timeseries = np.array(bs_timeseries)

    # K-Means clustering with DTW
    km_dtw = TimeSeriesKMeans(n_clusters=K, metric="dtw", verbose=False, random_state=42)

    # Perform clustering
    clusters = km_dtw.fit_predict(bs_timeseries)
    np.save(os.path.join(cluster_directory, f'clusters_{city}_{K}.npy'), clusters)
    # Calculate the DTW distance matrix
    dtw_distances = cdist_dtw(bs_timeseries)

    # Calculate the silhouette score using the precomputed DTW distances
    silhouette_avg = silhouette_score(dtw_distances, clusters, metric="precomputed")
    print(f'Silhouette Score with DTW: {silhouette_avg:.3f}')

    # Save the centroids
    centroid = km_dtw.cluster_centers_
    np.save(os.path.join(cluster_directory, f'centroid_{city}_{K}.npy'), centroid)
    # Log end time
    end_time = time.time()

    # Log total time taken for clustering
    total_time = end_time - start_time
    print(f"Time taken for clustering: {total_time:.2f} seconds")

    # Optionally, save the silhouette score and time to a log file
    with open(os.path.join(cluster_directory, f'log_{city}_{K}.txt'), 'a') as log_file:
        log_file.write(f'Silhouette Score: {silhouette_avg:.3f}\n')
        log_file.write(f'Time taken for clustering: {total_time:.2f} seconds\n')

    # Plot
    fig_rows = (K + 3) // 4  # Determine number of rows needed (4 plots per row)
    # Adjust figure size to accommodate multiple rows
    plt.figure(figsize=(16, 3 * fig_rows))  # Increase width and height for multiple rows

    for cluster_num in range(K):
        plt.subplot(fig_rows, 4, cluster_num + 1)  # Subplots with 4 per row
        for ts in bs_timeseries[clusters == cluster_num]:
            plt.plot(ts.flatten(), "k-", alpha=0.2)  # Plot each time series in the cluster with low opacity
        plt.plot(centroid[cluster_num].flatten(), "r-", linewidth=2)  # Plot the centroid with a red line
        plt.title(f"Cluster {cluster_num + 1}")

    plt.tight_layout()
    plt.savefig(os.path.join(cluster_directory, f'cluster_{city}_{K}.png'))

    # Count the number of BSs in each cluster
    unique, counts = np.unique(clusters, return_counts=True)
    # Generate different colors for each cluster
    colors = cm.get_cmap('tab20', len(unique))  # Using 'tab20' colormap for up to 20 clusters

    # Plot a bar plot with different colors for each cluster
    plt.figure(figsize=(7, 5))
    plt.bar(unique, counts, color=[colors(i) for i in range(len(unique))])
    plt.xlabel('Cluster Number')
    plt.ylabel('Number of BSs')
    plt.title(f'Number of Base Stations in Each Cluster ({city}, K={K})')
    plt.xticks(unique)  # Set x-axis ticks to match the cluster numbers
    plt.savefig(os.path.join(cluster_directory, f'cluster_count_{city}_{K}.png'))

    # Reshape the cluster labels into a 2D grid (nr x nr)
    cluster_grid = clusters.reshape(nr, nr)

    # Generate different colors for each cluster
    colors = cm.get_cmap('tab20', len(np.unique(clusters)))  # Using 'tab20' colormap for up to 20 clusters

    # Create an array to store the color for each BS based on its cluster
    color_grid = np.zeros((nr, nr, 3))  # For RGB values

    # Assign colors to the heatmap based on cluster
    for i in range(nr):
        for j in range(nr):
            color_grid[i, j] = colors(cluster_grid[i, j])[:3]  # Get RGB values from the colormap

    # Plot the heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(color_grid, aspect='auto')
    plt.title(f'Heatmap of Base Stations by Cluster ({city}, K={K})')
    plt.xticks([]) 
    plt.yticks([])
    plt.savefig(os.path.join(cluster_directory, f'cluster_heatmap_{city}_{K}.png'))

pair_list=[]
cities = ['Milan']
clusters = [3,4,6,8,10,15,20]

for city in cities:
    for K in clusters:
        pair_list.append((city, K))

if __name__ == '__main__':
    with Pool(2) as p:
        p.map(myMultiOpt,pair_list)