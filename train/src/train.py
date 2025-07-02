import os
import json
import glob
import logging
import argparse
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks # type: ignore
from bi_lstm_model import build_bilstm_classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from train_utils import load_config, save_config, pad_or_truncate_sequence, data_generator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
FEATURE_DIM_POSE_HANDS = (17 * 3) + (21 * 3 * 2)

DEFAULT_CONFIG = {
    'data_dir': 'processed_data',
    'output_model_dir': '../lstm/models',
    'log_file_name': 'training_log.csv',
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'sequence_length': 80,
    'feature_dim': FEATURE_DIM_POSE_HANDS,
    'hidden_size': 256,
    'num_layers': 2,
    'dropout_rate': 0.5,
    'test_size': 0.2,
    'validation_split': 0.2,
    'random_state': 42,
    'early_stopping_patience': 15,
    'lr_scheduler_factor': 0.5,
    'lr_scheduler_patience': 7,
}

def main():
    parser = argparse.ArgumentParser(description="Train a Bi-LSTM classifier on Pose and Hand data.")
    parser.add_argument('--config_file', type=str, default=None, help="Path to a JSON configuration file.")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.config_file:
        config.update(load_config(args.config_file))

    os.makedirs(config['output_model_dir'], exist_ok=True)
    save_config(config, os.path.join(config['output_model_dir'], f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"))

    logging.info(f"Loading data from: {config['data_dir']}")
    all_data_paths = glob.glob(os.path.join(config['data_dir'], '**', '*.npy'), recursive=True)
    if not all_data_paths:
        logging.error(f"No .npy files found in '{config['data_dir']}'. Exiting.")
        return

    unique_labels = sorted(list(set([path.split(os.sep)[-2] for path in all_data_paths])))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    num_classes = len(unique_labels)

    label_map_path = os.path.join(config['output_model_dir'], 'label_map.json')
    with open(label_map_path, 'w') as f:
        json.dump(idx_to_label, f, indent=4)
    logging.info(f"Label mapping saved to {label_map_path}")

    all_sequences, all_labels = [], []
    for path in all_data_paths:
        try:
            keypoints = np.load(path)
            if keypoints.shape[1] != config['feature_dim']:
                logging.warning(f"Feature dimension mismatch for {path}. Expected {config['feature_dim']}, got {keypoints.shape[1]}. Skipping.")
                continue
            all_sequences.append(pad_or_truncate_sequence(keypoints, config['sequence_length']))
            label_name = path.split(os.sep)[-2]
            all_labels.append(label_to_idx[label_name])
        except Exception as e:
            logging.error(f"Error loading or processing {path}: {e}. Skipping.")

    if not all_sequences:
        logging.error("No valid sequences found. Exiting.")
        return

    X = np.array(all_sequences, dtype=np.float32)
    y = tf.keras.utils.to_categorical(np.array(all_labels, dtype=np.int32), num_classes=num_classes)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=config['test_size'], random_state=config['random_state'], stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=config['validation_split'], random_state=config['random_state'], stratify=y_train_val)

    logging.info(f"TRAINING ON HANDS AND POSE ONLY. Feature dimension: {config['feature_dim']}")
    logging.info(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}, Testing samples: {X_test.shape[0]}")

    model = build_bilstm_classifier(
        input_shape=(config['sequence_length'], config['feature_dim']),
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_dim=num_classes,
        dropout_rate=config['dropout_rate']
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    model_callbacks = [
        callbacks.CSVLogger(os.path.join(config['output_model_dir'], config['log_file_name']), append=True),
        callbacks.EarlyStopping(monitor='val_loss', patience=config['early_stopping_patience'], restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=config['lr_scheduler_factor'], patience=config['lr_scheduler_patience']),
        callbacks.ModelCheckpoint(filepath=os.path.join(config['output_model_dir'], 'best_model_tf.keras'), monitor='val_loss', save_best_only=True)
    ]
    train_gen = data_generator(X_train, y_train, config['batch_size'], augment=True)
    val_gen = data_generator(X_val, y_val, config['batch_size'], augment=False)
    history = model.fit(train_gen, steps_per_epoch=max(1, len(X_train) // config['batch_size']), validation_data=val_gen, validation_steps=max(1, len(X_val) // config['batch_size']), epochs=config['epochs'], callbacks=model_callbacks, verbose=1)
    
    logging.info("Final Evaluation on Test Set")
    loss, accuracy, precision, recall = model.evaluate(X_test, y_test, batch_size=config['batch_size'], verbose=0)
    logging.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}, Test Precision: {precision:.4f}, Test Recall: {recall:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred_sparse = np.argmax(y_pred_probs, axis=1)
    y_test_sparse = np.argmax(y_test, axis=1)
    report = classification_report(y_test_sparse, y_pred_sparse, target_names=unique_labels, zero_division=0)
    logging.info(f"Classification Report:\n{report}")    
    cm = confusion_matrix(y_test_sparse, y_pred_sparse)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_model_dir'], 'confusion_matrix.png'))
    logging.info(f"Confusion matrix saved to {config['output_model_dir']}/confusion_matrix.png")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training and Validation Metrics', fontsize=16)
    metric_keys = ['loss', 'accuracy', 'precision', 'recall']
    for i, metric in enumerate(metric_keys):
        ax = axes[i//2, i%2]
        ax.plot(history.history[metric], label=f'Training {metric.capitalize()}')
        ax.plot(history.history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
        ax.set_title(f'Model {metric.capitalize()}')
        ax.set_ylabel(metric.capitalize())
        ax.set_xlabel('Epoch')
        ax.legend(loc='best')
        ax.grid(True)
    
    history_plot_path = os.path.join(config['output_model_dir'], 'training_history_metrics.png')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(history_plot_path)
    logging.info(f"Training history plots saved to {history_plot_path}")

if __name__ == "__main__":
    main()