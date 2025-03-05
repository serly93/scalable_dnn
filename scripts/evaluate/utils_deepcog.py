import pandas as pd
import glob
import os
import numpy as np
import statistics
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras import optimizers, activations, initializers, regularizers, constraints
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer, InputLayer, BatchNormalization, MaxPooling3D, Attention
from tensorflow.keras.utils import timeseries_dataset_from_array
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models, Input
import matplotlib
from matplotlib.colors import LogNorm
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error
from datetime import datetime, timezone,timedelta
from dateutil import tz
import tensorflow.keras.backend as K
from tensorflow import keras
import math
import statistics
import random
from IPython.display import display, HTML
from IPython.display import HTML
import random
import seaborn as sns
#tf.config.set_visible_devices([], 'GPU')
from IPython.display import display, HTML
import joblib
import time
import json
import csv

def evaluate_costs_single_clust(pred_load, real_load, traffic_peak, alpha):
    # Compute the difference between predicted and real load (error)
    error = pred_load - real_load
    # Compute overprovisioning and SLA violations
    tot_overprov = np.sum(error[np.where(error >= 0)])
    num_viol = len(error[np.where(error < 0)])
    sla_viol = np.multiply(traffic_peak, num_viol)
    sla_viol = np.multiply(sla_viol, alpha)
    # tot_overprov = np.array(tot_overprov, dtype = float)  
    tot_overprov = float(tot_overprov)   
    # Compute total cost
    total_cost = sla_viol + tot_overprov
    # service level agreement violation
    return total_cost, num_viol, tot_overprov

def evaluate_costs(pred_load, real_load, traffic_peak, alpha):
    # Compute the difference between predicted and real load (error)
    error = pred_load - real_load
    # Compute overprovisioning and SLA violations
    tot_overprov = np.sum(error[np.where(error >= 0)])
    num_viol = len(error[np.where(error < 0)])
    sla_viol = np.multiply(traffic_peak, num_viol)
    sla_viol = np.multiply(sla_viol, alpha)
    # tot_overprov = np.array(tot_overprov, dtype = float)  
    tot_overprov = float(tot_overprov)   
    # Compute total cost
    total_cost = sla_viol + tot_overprov
    # service level agreement violation
    return total_cost, sla_viol, num_viol, tot_overprov

def compute_mae_slanum(pred_load, real_load):
    error = pred_load - real_load
    num_viol = len(error[np.where(error < 0)])
    mae = np.mean(np.abs(error))
    return mae, num_viol
def compute_mae(pred_load, real_load):
    error = pred_load - real_load
    mae = np.mean(np.abs(error))
    return mae
# alpha-OMC loss function
def cost_func_more_args(alpha):
    def cost_func(y_true, y_pred):
        epsilon = 0.1
        diff = y_pred - y_true
        cost = np.zeros(diff.shape[0])
        # cost = np.zeros((diff.shape[0], diff.shape[1])) ## TO DO
        y1 = -epsilon * diff + alpha
        y2 = -np.true_divide(1, epsilon) * diff + alpha
        # y3 = np.true_divide(alpha, 1-(epsilon*alpha)) * (diff - (epsilon*alpha)) 
        y3 = -epsilon * alpha + diff 
        cost = tf.where(diff > (epsilon*alpha), y3, cost)
        cost = tf.where(diff < 0, y1, cost)
        cost = tf.where(tf.logical_and((diff <= (epsilon*alpha)), (diff >= 0)), y2, cost)
        cost = K.mean(cost, axis=-1)
        return cost
    return cost_func

# Generate neighboring cell ids for Milan dataset
def get_rows_Milan(cell_id, nr):
    row2 = []
    row3 = []

    for i in range(1,nr+1):
        if i> math.ceil(nr/2):
            globals()["row_%d" %i] = np.arange(100 * (i-math.ceil(nr/2)) + cell_id - math.floor(nr/2),100 * (i-math.ceil(nr/2)) + cell_id + math.floor(nr/2)+1)
        elif i<math.ceil(nr/2):
             globals()["row_%d" %i] = np.arange(cell_id - math.floor(nr/2) - 100 * (math.ceil(nr/2)-i) , cell_id + math.floor(nr/2) + 1 - 100 * (math.ceil(nr/2)-i))
        else:
             globals()["row_%d" %(math.ceil(nr/2))] = np.arange(cell_id - math.floor(nr/2), cell_id + math.floor(nr/2)+1)

    for j in range(1,nr+1):
        row1=globals()["row_%d" %j]
        #print(roww)
        row2= np.vstack(row1)
        row3=np.append(row3,row2).astype(int)
    return row3

def mae(y_true, y_pred):
    #y_true, predictions = np.array(y_true), np.array(predictions)
    #mae = np.mean(np.abs(predictions - y_true), axis = -1)
    #mean(abs(y_true - y_pred), axis=-1)
    mae = K.mean(K.abs(y_true - y_pred), axis = -1)
    return mae

def model_lstm(lookback, num_input, num_output):
    """Creates a model for time series forecasting with num_input cells as input and num_output cells as output."""
    model = models.Sequential()
    model.add(layers.LSTM(50, activation='tanh', input_shape=(lookback, num_input)))
    model.add(layers.Dense(num_output))  # Output one value (next time step for a specific cell)

    loss_function = 'mae' 
    optimizer = Adam(learning_rate=0.0005, beta_1=0.85, beta_2=0.98)
    model.compile(optimizer=optimizer, loss=loss_function)
    return model

def load_model_custom(city, CELL, alpha):
    model_path = f'../../../../oracle-data/serly/TMC_data/Trained_models/{city}/traffic_forecasting/Models/mymodel_{CELL}.h5' #TO DO change the cell for multiple cells
    loss_func = mae
    nn = tf.keras.models.load_model(model_path, custom_objects={'loss': loss_func}, compile=False)
    return nn

def load_test_data(city, cell):
    " loads the localized 5x5 test data for the given city and center cell"
    test_X = np.load(f'../../../../oracle-data/serly/TMC_data/Trained_models/{city}/Data_reshaped/test_{cell}.npy') #TO DO change the cell for multiple cells
    test_Y = np.load(f'../../../../oracle-data/serly/TMC_data/Trained_models/{city}/Data_reshaped/test_{cell}_Y.npy') #TO DO change the cell for multiple cells
    return test_X, test_Y

def model_deepcog(size, lookback, num_cluster, neurons, ker_sz):
    '''Build DeepCog architecture'''
    inputs = tf.keras.layers.Input(shape=(
        lookback,  size, size,  1))
    x = tf.keras.layers.Conv3D(neurons, kernel_size=(ker_sz, ker_sz, ker_sz), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.Conv3D(neurons, kernel_size=(ker_sz * 2, ker_sz * 2, ker_sz * 2), activation='relu', padding='same')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Conv3D(neurons / 2, kernel_size=(ker_sz * 2,ker_sz * 2, ker_sz * 2), activation='relu', padding='same')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(neurons * 2, activation='relu')(x)
    x = tf.keras.layers.Dense(neurons, activation='relu')(x)
    output = tf.keras.layers.Dense(num_cluster)(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(0.0005), loss = mae)
    return model

def load_model_deepcog(model_path):
    " loads the deepcog model with the custom loss function"

    loss_func = mae
    nn = tf.keras.models.load_model(model_path, custom_objects={'cost_func': loss_func}, compile=False)
    return nn

def preprocess_cluster_data(cluster_cells, all_cells, output_directory, city, lookback, nr, data_type="train"):
    """Prepares input sequences and targets with a lookback for each cluster,
    with a subset of cells (cluster_cells) as input and all cells in the cluster as targets.
    Returns the MinMaxScaler objects for cells in y to enable unnormalizing predictions.
    """
    X, y = [], []
    y_scalers = {}  # Dictionary to store scalers for the target cells

    # Load and normalize time series data for each cell in the cluster
    cell_data_list = {}
    for cell in all_cells:
        # Load the time series data for the cell (train or test depending on data_type)
        file_path = os.path.join(output_directory, f'{city}/{data_type}_{cell}.npy')
        ts_data = np.load(file_path)

        # Normalize the time series data using MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        ts_data = ts_data.reshape(-1, 1)  # Reshape to 2D array for the scaler
        ts_data_normalized = scaler.fit_transform(ts_data).flatten()

        # Save the scaler for later use
        y_scalers[cell] = scaler

        cell_data_list[cell] = ts_data_normalized

    # Create input sequences and the corresponding target values for each time series
    combined_data = np.stack([cell_data_list[cell] for cell in all_cells], axis=1)  # Shape: (time_steps, num_cells)
    input_data = np.stack([cell_data_list[cell] for cell in cluster_cells], axis=1)  # Shape: (time_steps, num_input_cells)

    for i in range(combined_data.shape[0] - lookback):
        # Extract input sequences for the subset of cells
        input_sequence = input_data[i:i + lookback]  # Shape: (lookback, num_input_cells)
        X.append(input_sequence)  # Input is shared across the subset of cells

        # Extract the target output, which is all cells at the next time step
        y.append(combined_data[i + lookback])  # All cell values as targets

    # Convert lists to numpy arrays
    X = np.array(X)  # Shape: (num_samples, lookback, num_input_cells)
    y = np.array(y)  # Shape: (num_samples, num_cells)
    X = X.reshape(X.shape[0], X.shape[1], nr, nr)  # Shape: (num_samples, lookback, nr, nr)

    # Reshape X for the DeepCOG model: (samples, lookback, features)
    # print(f"X shape: {X.shape}, y shape: {y.shape}")

    return X, y, y_scalers

def preprocess_cluster_data_lstm(cluster_cells, all_cells, output_directory, city, lookback, data_type="train"):
    """Prepares input sequences and targets with a lookback for each cluster,
    with a subset of cells (cluster_cells) as input and all cells in the cluster as targets.
    Returns the MinMaxScaler objects for cells in y to enable unnormalizing predictions.
    """
    X, y = [], []
    y_scalers = {}  # Dictionary to store scalers for the target cells

    # Load and normalize time series data for each cell in the cluster
    cell_data_list = {}
    for cell in all_cells:
        # Load the time series data for the cell (train or test depending on data_type)
        file_path = os.path.join(output_directory, f'{city}/{data_type}_{cell}.npy')
        ts_data = np.load(file_path)

        # Normalize the time series data using MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        ts_data = ts_data.reshape(-1, 1)  # Reshape to 2D array for the scaler
        ts_data_normalized = scaler.fit_transform(ts_data).flatten()

        # Save the scaler for later use
        y_scalers[cell] = scaler

        cell_data_list[cell] = ts_data_normalized

    # Create input sequences and the corresponding target values for each time series
    combined_data = np.stack([cell_data_list[cell] for cell in all_cells], axis=1)  # Shape: (time_steps, num_cells)
    input_data = np.stack([cell_data_list[cell] for cell in cluster_cells], axis=1)  # Shape: (time_steps, num_input_cells)

    for i in range(combined_data.shape[0] - lookback):
        # Extract input sequences for the subset of cells
        input_sequence = input_data[i:i + lookback]  # Shape: (lookback, num_input_cells)
        X.append(input_sequence)  # Input is shared across the subset of cells

        # Extract the target output, which is all cells at the next time step
        y.append(combined_data[i + lookback])  # All cell values as targets

    # Convert lists to numpy arrays
    X = np.array(X)  # Shape: (num_samples, lookback, num_input_cells)
    y = np.array(y)  # Shape: (num_samples, num_cells)

    # Reshape X for the DeepCOG model: (samples, lookback, features)
    # print(f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y, y_scalers

def preprocess_data(cell, output_directory, city, lookback, data_type="train"):
    """
    Preprocesses the time series data for a cell, applying normalization and 
    constructing the input X with the given lookback window.

    Args:
        cell (int): Identifier for the cell.
        output_directory (str): Directory where the data is stored.
        city (str): City name to structure the path.
        lookback (int): Number of past time steps to include as input features.
        nr (int): Not used but can be removed or modified as needed.
        data_type (str): Either "train" or "test" to specify data type.

    Returns:
        X (np.ndarray): Preprocessed inputs with shape (n_samples, lookback, 1).
        y (np.ndarray): Corresponding targets with shape (n_samples, 1).
        scaler (MinMaxScaler): Fitted scaler object for inverse normalization.
    """
    X, y = [], []

    # Load the time series data for the cell (train or test depending on data_type)
    file_path = os.path.join(output_directory, f'{city}/{data_type}_{cell}.npy')
    ts_data = np.load(file_path)

    # Normalize the time series data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    ts_data = ts_data.reshape(-1, 1)  # Reshape to 2D array for the scaler
    ts_data_normalized = scaler.fit_transform(ts_data).flatten()

    # Create the lookback data
    for t in range(len(ts_data_normalized) - lookback):
        X.append(ts_data_normalized[t:t + lookback])
        y.append(ts_data_normalized[t + lookback])

    X = np.array(X).reshape(-1, lookback, 1)  # Reshape to (n_samples, lookback, 1)
    y = np.array(y).reshape(-1, 1)  # Reshape to (n_samples, 1)

    return X, y, scaler

# Helper function to calculate Manhattan distance between two cells using their row and column indices
def manhattan_distance(cells, nr, cell1, cell2):
    idx1 = np.where(cells == cell1)[0][0]
    idx2 = np.where(cells == cell2)[0][0]

    row1, col1 = idx1 // nr, idx1 % nr
    row2, col2 = idx2 // nr, idx2 % nr

    return abs(row1 - row2) + abs(col1 - col2)

# Helper function to calculate the closest cells
def get_closest_cells(cells, nr, target_cell, cluster_cells, num_cells):
    # Find the num_cells closest cells to the target_cell within the cluster_cells
    distances = []
    for cell in cluster_cells:
        distance = manhattan_distance(cells, nr, target_cell, cell)
        distances.append((cell, distance))
    
    # Sort cells by distance
    distances.sort(key=lambda x: x[1])
    
    # Get the closest `num_cells`
    closest_cells = [cell for cell, dist in distances[:num_cells]]
    
    return closest_cells

def compute_local_nr2(K, nr2, city, nr, target_cluster_label):
    # Load the cells based on city
    cells = get_rows_Milan(5060, nr)
    cluster_counts = np.load(f'../../../../oracle-data/serly/MoE_data/cluster/clusters_{city}_{K}.npy')
    clustered_cells = {i: [] for i in range(K)}
    
    # Populate the cluster cells dictionary
    for idx, cluster_label in enumerate(cluster_counts):
        cell_id = cells[idx]
        clustered_cells[cluster_label].append(cell_id)

    # Check if target cluster label exists
    if target_cluster_label not in clustered_cells:
        print(f"Cluster label {target_cluster_label} not found.")
        return None

    # Compute local_nr2 for the specified cluster label
    cluster_cells = clustered_cells[target_cluster_label]
    local_nr2 = nr2
    local_num_cells = nr2 * nr2
    total_cells = len(cluster_cells)

    # Adjust if the number of cells in the cluster is less than local_num_cells
    if local_num_cells > total_cells:
        largest_square = int(math.floor(math.sqrt(total_cells))) ** 2
        local_num_cells = largest_square
        local_nr2 = int(math.sqrt(local_num_cells))

    # Print and return the local_nr2 for the target cluster label
    # print(f"Cluster label {target_cluster_label}: local_nr2 = {local_nr2}, K = {K}")
    return local_nr2
