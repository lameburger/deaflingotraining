import os
import numpy as np
from tensorflow.keras.utils import to_categorical

# Configuration
input_dir = 'completed_landmarks'  # Folder with subfolders for each sign (e.g. mom, dad, etc.)
output_X = 'X_data.npy'
output_y = 'y_data.npy'
target_timesteps = 50   # Fixed number of timesteps for LSTM input
num_features = 126      # 2 hands x 63 features per hand

# Get the list of labels (subfolder names)
labels = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
label_to_index = {label: idx for idx, label in enumerate(labels)}

all_sequences = []
all_labels = []

# Loop over each label/subfolder
for label in labels:
    label_folder = os.path.join(input_dir, label)
    # Loop over each .npy file (video sequence) in the folder.
    for filename in os.listdir(label_folder):
        if filename.endswith('.npy'):
            filepath = os.path.join(label_folder, filename)
            # Load the sequence. Expected shape: (num_frames, 2, 63)
            seq = np.load(filepath)
            # Flatten each frame to shape (126,)
            # If seq has shape (num_frames, 2, 63), then reshape to (num_frames, 126)
            seq = seq.reshape(seq.shape[0], -1)
            
            # Pad or crop the sequence to target_timesteps
            if seq.shape[0] < target_timesteps:
                # Pad with zeros at the end.
                padding = np.zeros((target_timesteps - seq.shape[0], num_features))
                seq = np.vstack([seq, padding])
            elif seq.shape[0] > target_timesteps:
                # Crop to the first target_timesteps frames.
                seq = seq[:target_timesteps]
            
            all_sequences.append(seq)
            all_labels.append(label_to_index[label])

# Convert to NumPy arrays.
X_data = np.array(all_sequences)   # Shape: (num_samples, target_timesteps, num_features)
y_data = np.array(all_labels)        # Shape: (num_samples,)

# Optionally, one-hot encode the labels if you're using categorical_crossentropy.
num_classes = len(labels)
y_data = to_categorical(y_data, num_classes)

# Save preprocessed data.
np.save(output_X, X_data)
np.save(output_y, y_data)

print(f"Preprocessed {len(all_sequences)} samples.")
print(f"X_data shape: {X_data.shape}")
print(f"y_data shape: {y_data.shape}")
