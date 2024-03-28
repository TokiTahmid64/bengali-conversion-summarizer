from src.transcribe.diarize import diarize_audio
from src.transcribe.transcribe import transcribe_audiofiles
import sys
import shutil

if __name__ == "__main__":
    audio_path, model_path = sys.argv[1], sys.argv[2]

    diarized_files = diarize_audio(audio_path)
    conversation = transcribe_audiofiles(diarized_files, model_path)

    shutil.rmtree("dia_output")

    conv_str = "\n".join([f"{name}: {text}" for name, text in conversation])
    with open("output.txt", "w") as f:
        f.write(conv_str)
