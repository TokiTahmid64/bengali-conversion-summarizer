# Bengali Conversation Summarizer

```bash
Record -> (Diarize -> Transcribe) -> Summarize
```

## How to use

Currently this repo consists only of the transcription part. An example audio is given, namely `talkshow.wav`. You need a GPU to run the transcription in this implementation. Change device in `src/transcribe/transcribe.py` to CPU if you don't have a GPU.

- Create a `.env` file in the root directory containing your huggingface token:

```bash
HF_TOKEN=<your_huggingface_token>
````

- You have to accept the terms from `pyannote` in huggingface. Go to their [model page](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the terms.

- Install the required packages:

```bash
pip install -r requirements.txt
```

- Download the necessary models from kaggle. Link: [https://www.kaggle.com/datasets/tugstugi/bengali-ai-asr-submission](https://www.kaggle.com/datasets/tugstugi/bengali-ai-asr-submission)

- To execute the diarized transcription pipeline, run the following command:

```bash
python main.py <path_to_audio_file> <directory_of_downloaded_models>
```
