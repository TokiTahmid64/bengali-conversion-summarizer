# Bengali Conversation Summarizer

This is a simple Bengali conversation summarizer. It takes a conversation audio as input and returns a summary text of the conversation, saved as `summary.txt`.

## Run

1. Download the required models from kaggle. You can download the models from [here](https://www.kaggle.com/datasets/tugstugi/bengali-ai-asr-submission).

2. You need to install the dependencies. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

3. You need to create an `.env` file in the root directory of the project and add the following environment variables:

```bash
HF_TOKEN=<your_hugging_face_token>
GOOGLE_API_KEY=<your_google_api_key>
```

**Note:** It is important that your huggingface account has access ``pyannote`. Go to their [model page](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the terms.

4. You can run the summarizer on your conversation audio by running the following command:

```bash
python main.py <path_to_audio_file> <directory_of_downloaded_models>
```

5. Summary will be available in `summary.txt` file.

## Demo

You can find a demo of the summarizer [here](https://www.kaggle.com/code/salmankhondker/bengali-conversation-summarizer-demo).
