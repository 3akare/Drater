# Drater
A multi-modal system for real-time sign language translation

## Build
1. Install Docker.
2. Clone the repository: `git clone https://github.com/3akare/Drater.git`
3. Navigate to the directory: `cd Drater`

**Note:**
* Create a `.env` file in the root directory (`Drater/.env`) and another in the `nlp` folder (`Drater/nlp/.env`).
* In both `.env` files, store your Gemini API key as follows: `GEMINI_API_KEY="**************"`
* Download the latest model (`.keras`) from the repository's releases section and place it in `lstm/models/`.
* In the `lstm/models/` directory, create a `label_map.json` file and paste the JSON labels as stated in the model's release notes.

4. Run `docker-compose up --build`
5. Visit `http://localhost:80` to use the application.

## Available Gestures
This is dependent on the model being used. Check the model's release notes for more details.

[View Drater's Interactive Report](https://3akare.github.io/Drater/)