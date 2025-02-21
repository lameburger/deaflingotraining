import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks, optimizers, models

# -------------------------------
# Configuration
# -------------------------------
data_dir = 'completed_landmarks'  # Root folder with subfolders per class (e.g., mom, dad, etc.)
target_timesteps = 50             # Fixed number of timesteps per sequence
num_features = 126                # Each frame is flattened to 126 features (2 x 63)
num_classes = 9                   # Update: 9 classes instead of 250
batch_size = 256

# -------------------------------
# Collect File Paths and Labels
# -------------------------------
file_paths = []
labels = []

label_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
label_to_index = {label: idx for idx, label in enumerate(label_names)}

for label in label_names:
    folder = os.path.join(data_dir, label)
    for fname in os.listdir(folder):
        if fname.endswith('.npy'):
            file_paths.append(os.path.join(folder, fname))
            labels.append(label_to_index[label])

file_paths = np.array(file_paths)
labels = np.array(labels)

print(f"Found {len(file_paths)} samples across labels: {label_names}")

# Shuffle the dataset
indices = np.arange(len(file_paths))
np.random.shuffle(indices)
file_paths = file_paths[indices]
labels = labels[indices]

# -------------------------------
# Split Data into Train, Validation, Test
# -------------------------------
total_samples = len(file_paths)
train_count = int(0.7 * total_samples)
val_count = int(0.15 * total_samples)

train_file_paths = file_paths[:train_count]
train_labels = labels[:train_count]
val_file_paths = file_paths[train_count:train_count + val_count]
val_labels = labels[train_count:train_count + val_count]
test_file_paths = file_paths[train_count + val_count:]
test_labels = labels[train_count + val_count:]

print(f"Train samples: {len(train_file_paths)}, Validation samples: {len(val_file_paths)}, Test samples: {len(test_file_paths)}")

# -------------------------------
# Create tf.data Dataset
# -------------------------------
def load_npy(path, label):
    # Convert byte string to regular string.
    path_str = path.decode('utf-8')
    # Load the npy file; expected shape: (num_frames, 2, 63) or (num_frames, 126)
    seq = np.load(path_str)
    # If the sequence is in shape (num_frames, 2, 63), flatten each frame to (126,)
    if len(seq.shape) == 3:
        seq = seq.reshape(seq.shape[0], -1)
    # Pad or crop to target_timesteps.
    T = seq.shape[0]
    if T < target_timesteps:
        pad = np.zeros((target_timesteps - T, seq.shape[1]), dtype=seq.dtype)
        seq = np.vstack([seq, pad])
    elif T > target_timesteps:
        seq = seq[:target_timesteps]
    return seq.astype(np.float32), np.int64(label)

def load_and_process(path, label):
    seq, lbl = tf.numpy_function(func=load_npy,
                                   inp=[path, label],
                                   Tout=[tf.float32, tf.int64])
    # Set static shape for downstream processing.
    seq.set_shape([target_timesteps, num_features])
    lbl.set_shape([])
    return seq, lbl

def create_dataset(file_paths, labels, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_dataset = create_dataset(train_file_paths, train_labels, batch_size, shuffle=True)
val_dataset = create_dataset(val_file_paths, val_labels, batch_size)
test_dataset = create_dataset(test_file_paths, test_labels, batch_size)

# -------------------------------
# Build the LSTM Model
# -------------------------------
model = models.Sequential([
    layers.Masking(mask_value=0.0, input_shape=(target_timesteps, num_features)),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# -------------------------------
# Callbacks
# -------------------------------
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint_cb = callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

# -------------------------------
# Train the Model
# -------------------------------
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=40,
                    callbacks=[early_stop, checkpoint_cb])

# -------------------------------
# Evaluate the Model
# -------------------------------
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
