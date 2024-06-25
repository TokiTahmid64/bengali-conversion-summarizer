from src.transcribe.diarize import diarize_audio
from src.transcribe.transcribe import transcribe_audiofiles
import sys
import shutil

if __name__ == "__main__":
    audio_path, model_path = sys.argv[1], sys.argv[2]

    diarized_files = diarize_audio(audio_path)
    conversation = transcribe_audiofiles(diarized_files, model_path)
    conv_str = "\n".join([f"{name}: {text}" for name, text in conversation])


    # save the conversation in a file. Please rename the file to the name of the audio file

    with open("{audio_path}.txt", "w") as f:
        f.write(conv_str)




    # # delete the diarized files
    # shutil.rmtree("dia_output")

    # # save the summary
    # with open("summary.txt", "w") as f:
    #     f.write(summary)
