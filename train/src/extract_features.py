import os
import cv2
import glob
import time
import logging
import argparse
import numpy as np
import mediapipe as mp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s %(message)s')

POSE_LANDMARKS_TO_EXTRACT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
NUM_POSE_FEATURES = len(POSE_LANDMARKS_TO_EXTRACT) * 3
NUM_HAND_FEATURES = 21 * 3
TOTAL_FEATURES = NUM_POSE_FEATURES + (NUM_HAND_FEATURES * 2)
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

def extract_combined_features(results_hands, results_pose) -> np.ndarray:
    """
    Extracts, normalizes, and combines features from Pose and Hands models.
    """
    pose_kps = np.zeros(NUM_POSE_FEATURES, dtype=np.float32)
    lhand_kps = np.zeros(NUM_HAND_FEATURES, dtype=np.float32)
    rhand_kps = np.zeros(NUM_HAND_FEATURES, dtype=np.float32)

    if results_pose.pose_landmarks:
        pose_landmarks = results_pose.pose_landmarks.landmark
        ref_point = pose_landmarks[mp_pose.PoseLandmark.NOSE.value]
        ref_x, ref_y, ref_z = ref_point.x, ref_point.y, ref_point.z
        pose_coords = []
        for i in POSE_LANDMARKS_TO_EXTRACT:
            lm = pose_landmarks[i]
            pose_coords.extend([lm.x - ref_x, lm.y - ref_y, lm.z - ref_z])
        pose_kps = np.array(pose_coords, dtype=np.float32)
    
    if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
        for i, landmarks in enumerate(results_hands.multi_hand_landmarks):
            handedness = results_hands.multi_handedness[i].classification[0].label
            wrist_lm = landmarks.landmark[0]
            wrist_x, wrist_y, wrist_z = wrist_lm.x, wrist_lm.y, wrist_lm.z
            hand_coords = []
            for lm in landmarks.landmark:
                hand_coords.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
            if handedness == "Left":
                lhand_kps = np.array(hand_coords, dtype=np.float32)
            elif handedness == "Right":
                rhand_kps = np.array(hand_coords, dtype=np.float32)
    return np.concatenate([pose_kps, lhand_kps, rhand_kps])

def process_video(video_path: str, output_filepath: str):
    """
    Processes a video, extracts combined keypoints, and saves the filtered sequence.
    """
    logging.info(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file {video_path}")
        return False

    valid_frames = []
    with mp_hands.Hands(min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE, max_num_hands=2) as hands, \
        mp_pose.Pose(min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_hands = hands.process(rgb_frame)
            results_pose = pose.process(rgb_frame)
            
            combined_keypoints = extract_combined_features(results_hands, results_pose)
            
            if combined_keypoints.any():
                valid_frames.append(combined_keypoints)
    cap.release()

    if not valid_frames:
        logging.warning(f"No valid frames with detected keypoints found in {video_path}. Skipping.")
        return False

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    try:
        np.save(output_filepath, np.array(valid_frames, dtype=np.float32))
        logging.info(f"Saved {len(valid_frames)} valid frames to {output_filepath}")
        return True
    except Exception as e:
        logging.error(f"Error saving keypoints to {output_filepath}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Extract keypoints.")
    parser.add_argument('--input_dir', type=str, default='raw_videos')
    parser.add_argument('--output_dir', type=str, default='processed_data')
    parser.add_argument('--force', action='store_true', help='Force reprocessing of all videos.')
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
        label = video_file.split(os.sep)[-2]
        filename = os.path.basename(video_file)
        name_without_ext = os.path.splitext(filename)[0]
        output_filepath = os.path.join(args.output_dir, label, f"{name_without_ext}.npy")

        if not args.force and os.path.exists(output_filepath):
            logging.info(f"Output file exists. Skipping: {output_filepath}")
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