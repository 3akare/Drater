import os
import sys
import logging
import textwrap
from google import genai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def refine_text(text: str) -> str:
    logging.info(f"Received text to refine: {text}")

    prompt_text = textwrap.dedent(f"""
    You are an exceptionally precise and powerful language processor.
    Your sole function is to transform fragmented or keyword-based inputs, representing detected ASL signs, into perfectly natural, grammatically correct, and coherent English sentences, statements, or short phrases.
    Ensure the output is concise and directly reflects the meaning of the signs. Avoid adding any conversational fluff or explanations about your process.
    Input: {text}
    Refined Output:""")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("GEMINI_API_KEY environment variable not set.")
        return ""

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt_text
        )
        refined_text = response.text.strip()
        logging.info(f"Refined text: {refined_text}")
        return refined_text
    except Exception as e:
        logging.error(f"Error refining text via Gemini API: {e}", exc_info=True)
        return ""
