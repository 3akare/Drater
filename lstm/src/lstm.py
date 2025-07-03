import os
import sys
import json
import grpc
import logging
import numpy as np
import tensorflow as tf
from concurrent import futures
from utils import pad_or_truncate_sequence
import prediction_services_pb2
import prediction_services_pb2_grpc

MAX_MESSAGE_LENGTH = 1024 * 1024 * 50
MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR, '../models', 'best_model_tf.keras')
LABEL_MAP_PATH = os.path.join(MODEL_DIR, '../models', 'label_map.json')

FEATURE_DIM = (17 * 3) + (21 * 3 * 2)
SEQUENCE_LENGTH = 80

WINDOW_SIZE = 30 
WINDOW_STRIDE = 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Model and Label Map
try:
    with open(LABEL_MAP_PATH, 'r') as f:
        idx_to_label_str_keys = json.load(f)
        IDX_TO_LABEL = {int(k): v for k, v in idx_to_label_str_keys.items()}
    model = tf.keras.models.load_model(MODEL_PATH)
    dummy_input = np.zeros((1, SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
    model.predict(dummy_input, verbose=0)
    logging.info(f"Successfully loaded model and label map for {FEATURE_DIM}-dimensional input.")
except Exception as e:
    logging.critical(f"FATAL ERROR: Could not load resources. Exiting. Error: {e}", exc_info=True)
    sys.exit(1)

class LstmPredictionService(prediction_services_pb2_grpc.LstmServiceServicer):
    def Predict(self, request, context):
        """
        Predicts a sequence of gestures from a continuous stream of keypoints
        using a sliding window and a model trained with a '_blank_' class.
        """
        try:
            if not request.gestures:
                context.set_details("No gestures provided.")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return prediction_services_pb2.LstmResponse()

            input_frames = [frame.keypoints for frame in request.gestures[0].frames]
            live_sequence_np = np.array(input_frames, dtype=np.float32)

            if live_sequence_np.ndim != 2 or live_sequence_np.shape[1] != FEATURE_DIM:
                details = f"Invalid keypoint dimension. Expected (N, {FEATURE_DIM}), got {live_sequence_np.shape}"
                context.set_details(details)
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return prediction_services_pb2.LstmResponse()

            raw_predictions = []
            total_frames = live_sequence_np.shape[0]

            for i in range(0, total_frames - WINDOW_SIZE + 1, WINDOW_STRIDE):
                window = live_sequence_np[i : i + WINDOW_SIZE, :]
                
                sequence_to_predict = pad_or_truncate_sequence(window, SEQUENCE_LENGTH)
                sequence_to_predict = np.expand_dims(sequence_to_predict, axis=0)
                
                prediction_probs = model.predict(sequence_to_predict, verbose=0)
                predicted_index = np.argmax(prediction_probs)
                predicted_label = IDX_TO_LABEL.get(predicted_index, "_blank_")

                if np.max(prediction_probs) > 0.6:
                    raw_predictions.append(predicted_label)
            
            final_sentence = []
            if raw_predictions:
                if raw_predictions[0] != '_blank_':
                    final_sentence.append(raw_predictions[0])
                
                for pred in raw_predictions[1:]:
                    if pred != '_blank_' and (not final_sentence or pred != final_sentence[-1]):
                        final_sentence.append(pred)

            response_text = " ".join(final_sentence)
            logging.info(f"Raw predictions: {raw_predictions}")
            logging.info(f"Final sentence: '{response_text}'")
            return prediction_services_pb2.LstmResponse(translated_text=response_text)

        except Exception as e:
            logging.error(f"Error in Predict RPC: {e}", exc_info=True)
            context.set_details(f"Internal server error: {type(e).__name__}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return prediction_services_pb2.LstmResponse()

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=(os.cpu_count() or 4)),
        options=[
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH)
        ]
    )
    prediction_services_pb2_grpc.add_LstmServiceServicer_to_server(LstmPredictionService(), server)
    server.add_insecure_port("[::]:50051")
    logging.info("LSTM gRPC Server started on port 50051.")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
