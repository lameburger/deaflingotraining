import os
import numpy as np
import tensorflow as tf

# Configuration: adjust these as needed.
data_dir = 'completed_landmarks'  # Root folder with subfolders for each word (e.g., mom, dad, etc.)
target_timesteps = 50             # Fixed number of timesteps for each sequence
num_features = 126                # Expected features per frame (2 * 63)
batch_size = 256

# Get a list of all npy file paths and corresponding labels.
file_paths = []
labels = []

# List all subfolders (each represents a label).
label_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
label_to_index = {name: i for i, name in enumerate(label_names)}

for label in label_names:
    folder = os.path.join(data_dir, label)
    for fname in os.listdir(folder):
        if fname.endswith('.npy'):
            file_paths.append(os.path.join(folder, fname))
            labels.append(label_to_index[label])

print(f"Found {len(file_paths)} samples across labels: {label_names}")

# Convert the lists to NumPy arrays.
file_paths = np.array(file_paths)
labels = np.array(labels)

# Create a tf.data Dataset from the file paths and labels.
dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

# Function to load and preprocess a single npy file.
def load_npy(path, label):
    """
    Loads a npy file, flattens frames if needed, and pads/crops the sequence.
    """
    path_str = path.decode('utf-8')
    seq = np.load(path_str)  # Expected shape: (num_frames, 2, 63) or (num_frames, 126)
    
    # If the sequence has 3 dimensions, flatten each frame to shape (126,)
    if len(seq.shape) == 3:
        seq = seq.reshape(seq.shape[0], -1)
    
    # Pad or crop the sequence to target_timesteps.
    T = seq.shape[0]
    if T < target_timesteps:
        pad_width = target_timesteps - T
        pad = np.zeros((pad_width, seq.shape[1]), dtype=seq.dtype)
        seq = np.vstack([seq, pad])
    elif T > target_timesteps:
        seq = seq[:target_timesteps]
    
    return seq.astype(np.float32), np.int64(label)

# Wrapper to use tf.numpy_function.
def load_and_process(path, label):
    seq, lbl = tf.numpy_function(
        func=load_npy,
        inp=[path, label],
        Tout=[tf.float32, tf.int64]
    )
    # Set the static shape information.
    seq.set_shape([target_timesteps, num_features])
    lbl.set_shape([])
    return seq, lbl

# Map the load_and_process function onto the dataset.
dataset = dataset.map(load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
# Shuffle, batch, and prefetch the dataset for optimal performance.
dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# At this point, `dataset` is ready to be fed to your model for training.
