# Import required libraries
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
    city = pair

    # Load cells based on the city
    cells = get_rows_Milan(5060, nr)


    model_dir = f'../../../../oracle-data/serly/Scalable_dnn/Trained_models/DeepCOG_original/{city}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

   
    # Train one model for each cluster with the input of shape (num_cells, lookback, 1) and output of shape (total_cells, 1)
    start_time = time.time()
    print(f"Training model for {city} with {nr} cells and {lookback} lookback.")
    # Preprocess the training data for the selected cells
    X_train, y_train, _ = preprocess_cluster_data(cells, cells, output_directory, city, lookback, nr, data_type="train")

    # Create the model with local_num_cells as input and total_cells as output
    model = model_deepcog(nr, lookback, nr*nr, neurons, ker_sz)

    # Train the model on the entire training dataset
    model.fit(X_train, y_train, epochs=epochs, batch_size=batchsize, verbose=0)

    # Save the model to disk
    model_filename = f"deepcog_size_{nr}.h5"
    model.save(os.path.join(model_dir, model_filename))

    end_time = time.time()
    training_duration = end_time - start_time

    # Check if the file already exists
    log_file_path = os.path.join(model_dir, f'time_log_size_{nr}.txt')
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'Training took {training_duration:.2f} seconds with size {nr}.\n')

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


# Create list of parameter pairs for multiprocessing
pair_list = []
cities = ['Milan']
for city in cities:
    pair_list.append((city))

if __name__ == '__main__':
    with Pool(30) as p:
        p.map(myMultiOpt, pair_list)
