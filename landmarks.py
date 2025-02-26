#!/usr/bin/env python
import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

def process_video(video_path, output_dir, rel_dir):
    """
    Process a single video:
      - Reads video frames.
      - Extracts hand landmarks for up to 2 hands per frame.
      - Pads with zeros if less than 2 hands are detected.
      - Saves the landmarks sequence as a .npy file.
    The output is saved to a subfolder corresponding to the relative directory (i.e., the sign).
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Create the output folder for this sign if it doesn't exist.
    output_subdir = os.path.join(output_dir, rel_dir)
    os.makedirs(output_subdir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    all_landmarks = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Create default two hands with zeros (each hand: 21 landmarks * 3 values).
            hands_landmarks = [[0] * (21 * 3) for _ in range(2)]
            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if i >= 2:
                        break
                    landmarks = ([lm.x for lm in hand_landmarks.landmark] +
                                 [lm.y for lm in hand_landmarks.landmark] +
                                 [lm.z for lm in hand_landmarks.landmark])
                    hands_landmarks[i] = landmarks

            all_landmarks.append(hands_landmarks)

    cap.release()
    all_landmarks_np = np.array(all_landmarks)  # Shape: (num_frames, 2, 63)
    output_path = os.path.join(output_subdir, f'{video_name}.npy')
    np.save(output_path, all_landmarks_np)
    # Uncomment for debugging:
    # print(f"Processed: {video_name} in folder: {rel_dir}")

def get_video_paths(input_dir, extensions=('.mp4', '.avi', '.mov')):
    """
    Walk through input_dir and gather video file paths.
    Returns a list of tuples: (full_video_path, relative_subfolder)
    """
    video_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(extensions):
                full_path = os.path.join(root, file)
                rel_dir = os.path.relpath(root, input_dir)
                video_paths.append((full_path, rel_dir))
    return video_paths

if __name__ == "__main__":
    # Directories: raw_data contains subfolders for each sign; raw_landmarks will mimic that structure.
    input_dir = "lesson8"
    output_dir = "lesson8_landmarks"
    os.makedirs(output_dir, exist_ok=True)

    video_list = get_video_paths(input_dir)
    total_videos = len(video_list)
    print(f"Total videos found: {total_videos}")

    # Determine an optimal batch size (you can adjust this heuristic)
    if total_videos < 100:
        BATCH_SIZE = 5
    elif total_videos < 1000:
        BATCH_SIZE = 10
    else:
        BATCH_SIZE = 20

    # Calculate number of batches needed.
    num_batches = math.ceil(total_videos / BATCH_SIZE)
    print(f"Using batch size: {BATCH_SIZE} => Total batches: {num_batches}")

    # Check if running as a SLURM array job.
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_id is not None:
        batch_index = int(array_id)
        start = batch_index * BATCH_SIZE
        end = start + BATCH_SIZE
        if start >= total_videos:
            print(f"Batch index {batch_index} is out of range. Exiting.")
        else:
            batch_videos = video_list[start:end]
            print(f"Processing batch {batch_index + 1}/{num_batches}: videos {start} to {min(end, total_videos)}")
            for video_path, rel_dir in tqdm(batch_videos, desc="Processing Batch Videos"):
                process_video(video_path, output_dir, rel_dir)
    else:
        # No SLURM array index; process all videos sequentially.
        for video_path, rel_dir in tqdm(video_list, desc="Processing Videos"):
            process_video(video_path, output_dir, rel_dir)
