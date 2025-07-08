import os
import glob
import logging
import argparse
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_feature_summary(data: np.ndarray) -> np.ndarray:
    """Creates a single feature vector summarizing a whole sequence."""
    # We can use mean and std deviation as a simple summary
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return np.concatenate([mean, std])

def visualize_data_clusters(data_dir: str):
    """
    Loads all processed data, runs t-SNE, and plots the 2D clusters.
    """
    logging.info(f"Loading all data from: {data_dir}")
    all_data_paths = glob.glob(os.path.join(data_dir, '**', '*.npy'), recursive=True)
    if not all_data_paths:
        logging.error(f"No .npy files found in '{data_dir}'. Exiting.")
        return

    all_summaries = []
    all_labels = []
    
    for path in all_data_paths:
        try:
            keypoints = np.load(path)
            if keypoints.size == 0: continue
            
            summary_vector = create_feature_summary(keypoints)
            all_summaries.append(summary_vector)
            
            label_name = path.split(os.sep)[-2]
            all_labels.append(label_name)
        except Exception as e:
            logging.error(f"Error processing {path}: {e}")

    if not all_summaries:
        logging.error("No valid data could be processed.")
        return

    feature_matrix = np.array(all_summaries)
    labels_array = np.array(all_labels)

    logging.info(f"Running t-SNE on {feature_matrix.shape[0]} samples... This may take a moment.")
    tsne = TSNE(n_components=2, verbose=1, perplexity=min(30, len(feature_matrix)-1), n_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(feature_matrix)

    logging.info("Plotting results...")
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        hue=labels_array,
        palette=sns.color_palette("hsv", len(np.unique(labels_array))),
        s=100,
        alpha=0.8
    )
    plt.title('t-SNE Cluster Visualization of Gesture Data')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(loc='best')
    plt.grid(True)
    
    print("\nClose the plot window to exit.")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize gesture data clusters using t-SNE.")
    parser.add_argument('--data_dir', type=str, default='processed_data', help="Directory containing the processed .npy files.")
    args = parser.parse_args()
    visualize_data_clusters(args.data_dir)