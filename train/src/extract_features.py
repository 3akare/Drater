import os
import cv2
import glob
import time
import logging
import argparse
import numpy as np
import mediapipe as mp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


mp_hands = mp.solutions.hands
NUM_HAND_FEATURES = 21 * 3  # 63 features per hand
TOTAL_FEATURES = NUM_HAND_FEATURES * 2 # 126 features total
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

def extract_hand_features(results) -> np.ndarray:
    """
    Extracts and normalizes features for left and right hands from Hands model results.
    """
    left_hand_kps = np.zeros(NUM_HAND_FEATURES, dtype=np.float32)
    right_hand_kps = np.zeros(NUM_HAND_FEATURES, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[i].classification[0].label
            wrist_lm = landmarks.landmark[0]
            wrist_x, wrist_y, wrist_z = wrist_lm.x, wrist_lm.y, wrist_lm.z
            hand_coords = []
            for lm in landmarks.landmark:
                hand_coords.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
            if handedness == "Left":
                left_hand_kps = np.array(hand_coords, dtype=np.float32)
            elif handedness == "Right":
                right_hand_kps = np.array(hand_coords, dtype=np.float32)
    return np.concatenate([left_hand_kps, right_hand_kps])

def process_video(video_path: str, output_filepath: str):
    logging.info(f"Processing video for hand keypoints: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file {video_path}")
        return False

    valid_frames = []
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            hand_keypoints = extract_hand_features(results)
            
            if hand_keypoints.any():
                valid_frames.append(hand_keypoints)
    cap.release()

    if not valid_frames:
        logging.warning(f"No valid hand keypoints found in {video_path}. Skipping.")
        return False
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    np.save(output_filepath, np.array(valid_frames, dtype=np.float32))
    logging.info(f"Saved {len(valid_frames)} valid frames to {output_filepath}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Extract Hand keypoints using the MediaPipe Hands model.")
    parser.add_argument('--input_dir', type=str, default='raw_videos')
    parser.add_argument('--output_dir', type=str, default='processed_data')
    parser.add_argument('--force', action='store_true', help="Force reprocessing of all videos.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    video_files = glob.glob(os.path.join(args.input_dir, '**', '*.mp4'), recursive=True)
    video_files.sort()

    if not video_files:
        logging.warning(f"No video files found in '{args.input_dir}'.")
        return
    logging.info(f"Found {len(video_files)} videos to process.")
    start_time = time.time()
    processed_count, failed_count, skipped_count = 0, 0, 0

    for video_file in video_files:
        label = os.path.basename(os.path.dirname(video_file))
        filename = os.path.basename(video_file)
        name_without_ext = os.path.splitext(filename)[0]
        output_filepath = os.path.join(args.output_dir, label, f"{name_without_ext}.npy")

        if not args.force and os.path.exists(output_filepath):
            skipped_count += 1
            continue

        if process_video(video_file, output_filepath):
            processed_count += 1
        else:
            failed_count += 1

    end_time = time.time()
    logging.info("Processing Complete")
    logging.info(f"Videos processed: {processed_count}, Skipped: {skipped_count}, Failed: {failed_count}")
    logging.info(f"Total time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()