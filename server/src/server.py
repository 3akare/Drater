import os
import grpc
import logging
from flask_cors import CORS
import prediction_services_pb2
import prediction_services_pb2_grpc
from flask import Flask, jsonify, request

app = Flask(__name__)
CORS(app)

# Configuration
DEBUG_MODE = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
LSTM_SERVICE_HOST = os.getenv("LSTM_HOST", "localhost:50051")
NLP_SERVICE_HOST = os.getenv("NLP_HOST", "localhost:50052")
app.config['DEBUG'] = DEBUG_MODE
MAX_GRPC_MESSAGE_LENGTH = 1024 * 1024 * 50

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@app.route("/predict_gesture", methods=["POST"])
def predict_from_keypoints():
    """
    Receives a JSON payload with a single gesture sequence (keypoints),
    gets a prediction from the LSTM service,
    refines it with the NLP service (if enabled), and returns the final text.
    """
    logging.info("Keypoint prediction request received.")
    try:
        data = request.get_json()
        if not data or 'keypoints' not in data:
            logging.warning("Prediction request is missing JSON data or 'keypoints' key.")
            return jsonify({"error": "Invalid request: Missing 'keypoints' in JSON body."}), 400
        single_gesture_keypoints_sequence = data['keypoints']

        if not isinstance(single_gesture_keypoints_sequence, list) or len(single_gesture_keypoints_sequence) == 0:
            logging.warning("Received empty or invalid gesture sequence.")
            return jsonify({"error": "Invalid keypoints data provided. Expected a list of frames for a single gesture."}), 422
        
        logging.info(f"Received 1 gesture sequence with {len(single_gesture_keypoints_sequence)} frames for prediction.")

    except Exception as e:
        logging.error(f"Error parsing request JSON: {e}", exc_info=True)
        return jsonify({"error": "Failed to parse request data."}), 400

    # LSTM gRPC Call
    lstm_translated_text = ""
    try:
        gestures_for_request = []
        frames_for_current_gesture = [
            prediction_services_pb2.KeypointFrame(keypoints=frame_data)
            for frame_data in single_gesture_keypoints_sequence
        ]
        
        gesture_seq_msg = prediction_services_pb2.GestureSequence(frames=frames_for_current_gesture)
        gestures_for_request.append(gesture_seq_msg)
        request_message_lstm = prediction_services_pb2.LstmRequest(gestures=gestures_for_request)
        
        logging.info(f"Sending {len(gestures_for_request)} gesture(s) to LSTM gRPC service at {LSTM_SERVICE_HOST}...")
        grpc_options = [
            ('grpc.max_receive_message_length', MAX_GRPC_MESSAGE_LENGTH),
            ('grpc.max_send_message_length', MAX_GRPC_MESSAGE_LENGTH)
        ]
        with grpc.insecure_channel(LSTM_SERVICE_HOST, options=grpc_options) as channel:
            stub_lstm = prediction_services_pb2_grpc.LstmServiceStub(channel)
            response_lstm = stub_lstm.Predict(request_message_lstm)
            lstm_translated_text = response_lstm.translated_text
            logging.info(f"Received translated text from LSTM: '{lstm_translated_text}'")

            if not lstm_translated_text.strip():
                 logging.warning("LSTM service returned an empty translation for a valid keypoint sequence.")

    except grpc.RpcError as rpc_e:
        logging.error(f"LSTM gRPC call failed: {rpc_e.details()} (Code: {rpc_e.code().name})", exc_info=True)
        return jsonify({"message": f"Sign language translation service failed: {rpc_e.code().name}"}), 503
    except Exception as e:
        logging.error(f"An unexpected error occurred during LSTM gRPC call: {e}", exc_info=True)
        return jsonify({"message": "An unexpected error occurred during sign language translation."}), 500

    final_nlp_response_text = lstm_translated_text
    if lstm_translated_text and lstm_translated_text.strip(): 
        try:
            logging.info(f"Sending translated text to NLP gRPC service at {NLP_SERVICE_HOST}...")
            request_message_nlp = prediction_services_pb2.NlpRequest(raw_text=lstm_translated_text)
            
            grpc_options_nlp = [
                ('grpc.max_receive_message_length', MAX_GRPC_MESSAGE_LENGTH),
                ('grpc.max_send_message_length', MAX_GRPC_MESSAGE_LENGTH)
            ]
            with grpc.insecure_channel(NLP_SERVICE_HOST, options=grpc_options_nlp) as channel:
                stub_nlp = prediction_services_pb2_grpc.NlpServiceStub(channel)
                response_nlp = stub_nlp.Refine(request_message_nlp)
                final_nlp_response_text = response_nlp.refined_text
                logging.info(f"Received refined text from NLP: '{final_nlp_response_text}'")
        except grpc.RpcError as rpc_e:
            logging.error(f"NLP gRPC call failed: {rpc_e.details()} (Code: {rpc_e.code().name})", exc_info=True)
            logging.warning("NLP service call failed. Returning raw LSTM translation.")
        except Exception as e:
            logging.error(f"An unexpected error occurred during NLP gRPC call: {e}", exc_info=True)
            logging.warning("Unexpected error during NLP call. Returning raw LSTM translation.")
    else:
        final_nlp_response_text = "Could not translate the detected gestures."

    logging.info("Request processing finished successfully.")
    return jsonify({"translatedText": final_nlp_response_text}), 200

if __name__ == '__main__':
    logging.info(f"Starting Flask server on 0.0.0.0:8080 (Debug: {DEBUG_MODE})")
    app.run(host="0.0.0.0", port=8080, debug=True)
