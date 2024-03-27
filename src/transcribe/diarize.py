from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core.annotation import Annotation
from pydub import AudioSegment
import os
import gc
from dotenv import load_dotenv

load_dotenv()


def diarize_audio(fpath: str) -> list[tuple[str, float, float, str]]:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=os.environ["HF_TOKEN"]
    )

    # get diarization
    with ProgressHook() as hook:
        diarization: Annotation = pipeline(fpath, hook=hook)

    # merge consecutive speech turns from the same speaker
    results: list[tuple] = []
    for speechturn, _, speaker in diarization.itertracks(yield_label=True):  # type: ignore
        speaker, start, end = str(speaker), speechturn.start, speechturn.end
        if results and results[-1][0] == speaker:
            results[-1] = (speaker, results[-1][1], end)  # update end time
        else:
            results.append((speaker, start, end))

    # slice each speaker turns from the audio file
    if not os.path.exists("dia_output"):
        os.makedirs("dia_output")

    audio = AudioSegment.from_wav(fpath)
    for i, (speaker, start, end) in enumerate(results):
        speaker_audio = audio[int(start * 1000) : int(end * 1000)]
        output_path = f"dia_output/{i}_{speaker}.wav"
        speaker_audio.export(output_path, format="wav")
        results[i] = (*results[i], output_path)

    # free memory used by model
    del pipeline, diarization, audio
    gc.collect()

    return results
