from src.transcribe.diarize import diarize_audio
from src.transcribe.transcribe import transcribe_audiofiles

if __name__ == "__main__":
    diarized_files = diarize_audio("./talkshow.wav")
    conversation = transcribe_audiofiles(diarized_files)
    print(conversation)
