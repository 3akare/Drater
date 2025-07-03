import os
import grpc
import logging
from concurrent import futures
from dotenv import load_dotenv
import prediction_services_pb2
from nlp_utils import refine_text
import prediction_services_pb2_grpc

load_dotenv()

# Configuration
MAX_MESSAGE_LENGTH = 1024 * 1024 * 50

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NlpPredictionService(prediction_services_pb2_grpc.LstmServiceServicer):
    def Refine(self, request, context):
        """
        Receives raw text representing detected ASL signs and returns
        refined, natural English text using Gemini API.
        """
        try:
            input_text = request.raw_text.strip()
            logging.info(f"Received text for NLP processing: \"{input_text}\"")
            if not input_text:
                logging.warning("NLP service received empty or whitespace-only request data.")
                return prediction_services_pb2.NlpResponse(refined_text="")
            response_text = refine_text(input_text)
            logging.info(f"NLP processed response: \"{response_text}\"")
            return prediction_services_pb2.NlpResponse(refined_text=response_text)
        except Exception as e:
            logging.error(f"Error in NLP biDirectionalStream: {e}", exc_info=True)
            context.set_details(f"Internal server error during NLP processing: {type(e).__name__}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return prediction_services_pb2.NlpResponse(refined_text="Error processing text.")

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 2 if os.cpu_count() else 4),
        options=[
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH)
        ]
    )
    
    prediction_services_pb2_grpc.add_NlpServiceServicer_to_server(NlpPredictionService(), server)
    server.add_insecure_port("[::]:50052")
    logging.info("NLP gRPC Server started on port 50052")
    try:
        server.start()
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("NLP gRPC Server shutting down.")
        server.stop(0)
    except Exception as e:
        logging.critical(f"NLP gRPC Server failed: {e}", exc_info=True)

if __name__ == "__main__":
    serve()
