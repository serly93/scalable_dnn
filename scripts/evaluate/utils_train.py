from scipy.spatial.distance import cdist
from functions.utils import *

# Check for neighboring BSs
def are_neighbors(bs1, bs2):
    """Check if two base stations are neighbors (horizontally or vertically adjacent)."""
    r1, c1 = bs1
    r2, c2 = bs2
    return (abs(r1 - r2) == 1 and c1 == c2) or (abs(c1 - c2) == 1 and r1 == r2)

def find_neighbors(bs_positions, start_idx, visited):
    """Find all connected neighbors (direct or indirect) for a given base station."""
    stack = [start_idx]  # Start with the current BS index
    group = []  # To store the group of connected BSs

    while stack:
        idx = stack.pop()
        if idx not in visited:
            visited.add(idx)
            group.append(bs_positions[idx])

            # Check all other BSs for neighbors and add them to the stack if they're connected
            for i, bs in enumerate(bs_positions):
                if i not in visited and are_neighbors(bs_positions[idx], bs):
                    stack.append(i)

    return group

def arrange_in_grid(neighbor_group):
    """Arrange the BSs in a rectangular grid, keeping neighbors together as much as possible."""
    num_bss = len(neighbor_group)
    
    # Determine the grid size (rows, cols) based on the number of BSs
    cols = math.ceil(math.sqrt(num_bss))  # Try to make a near-square grid
    rows = math.ceil(num_bss / cols)
    
    # Create a grid of None placeholders
    grid = np.full((rows, cols), None)
    
    # Fill the grid with the BSs from the group
    for i, bs in enumerate(neighbor_group):
        row = i // cols
        col = i % cols
        grid[row, col] = bs
    
    return grid

def preprocess_data(grid, output_directory, city, sequence_length):
    """ Preprocesses the grid and loads time series data from .npy files for model input. """
    X = []
    y = []
    bs_sequence_map = []  # To keep track of which BS each sequence belongs to
    
    # Iterate through each cell in the grid
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i, j] is not None:
                # Get the BS identifier (e.g., 'BS1', 'BS2', etc.)
                bs_id = grid[i, j]
                
                # Load the time series data for this specific BS from a .npy file
                file_path = os.path.join(output_directory, f'{city}/train_{bs_id}.npy')
                ts_data = np.load(file_path)
                scaler = MinMaxScaler()
                ts_data = scaler.fit_transform(ts_data.reshape(-1, 1)).flatten()
                # Prepare the input sequence (X) and target value (y)
                for k in range(len(ts_data) - sequence_length):
                    X.append(ts_data[k:k+sequence_length])
                    y.append(ts_data[k+sequence_length])
                    bs_sequence_map.append(bs_id)  # Store which BS this sequence comes from
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X for LSTM input: (samples, timesteps, features)
    X = X.reshape((X.shape[0], sequence_length, 1))
    
    return X, y, bs_sequence_map