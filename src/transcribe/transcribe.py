import warnings
import transformers
import torch
from pathlib import Path

print(transformers.__version__)
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

MODEL = "bengali-whisper-medium"
PUNCT_MODELS = [
    "punct-model-6layers/",
    "punct-model-8layers/",
    "punct-model-11layers/",
    "punct-model-12layers/",
]

CHUNK_LENGTH_S = 20.1
PUNCT_WEIGHTS = [[1.0, 1.4, 1.0, 0.8]]
BATCH_SIZE = 4


def fix_repetition(text, max_count):
    uniq_word_counter = {}
    words = text.split()
    for word in text.split():
        if word not in uniq_word_counter:
            uniq_word_counter[word] = 1
        else:
            uniq_word_counter[word] += 1

    for word, count in uniq_word_counter.items():
        if count > max_count:
            words = [w for w in words if w != word]
    text = " ".join(words)
    return text


def punctuate(text, tokenizer, models):
    input_ids = tokenizer(text).input_ids
    with torch.no_grad():
        model = models[0]
        logits = torch.nn.functional.softmax(
            model(input_ids=torch.LongTensor([input_ids]).cuda()).logits[0, 1:-1], dim=1
        ).cpu()
        for model in models[1:]:
            logits += torch.nn.functional.softmax(
                model(input_ids=torch.LongTensor([input_ids]).cuda()).logits[0, 1:-1],
                dim=1,
            ).cpu()
        logits = logits / len(models)
        logits *= torch.FloatTensor(PUNCT_WEIGHTS)
        label_ids = torch.argmax(logits, dim=-1)

        tokens = tokenizer(text, add_special_tokens=False).input_ids
        punct_text = ""
        for index, token in enumerate(tokens):
            token_str = tokenizer.decode(token)
            if "##" not in token_str:
                punct_text += " " + token_str
            else:
                punct_text += token_str[2:]
            punct_text += ["", "।", ",", "?"][label_ids[index].item()]  # type: ignore

    punct_text = punct_text.strip()
    return punct_text


def transcribe_audiofiles(
    dia_results: list[tuple[str, float, float, str]], models_path: str
) -> list[tuple[str, str]]:
    # load models
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=str(Path(models_path) / MODEL),
        tokenizer=MODEL,
        chunk_length_s=CHUNK_LENGTH_S,
        device=0,
        batch_size=BATCH_SIZE,
    )
    pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language="bn", task="transcribe")  # type: ignore
    print("model loaded!")

    # transcribe
    results: list[tuple[str, str]] = []
    texts = pipe(
        [result[3] for result in dia_results],
        generate_kwargs={"max_length": 260, "num_beams": 4},
    )

    # clean transcription model
    del pipe
    torch.cuda.empty_cache()

    # load punctuation models
    models = [
        AutoModelForTokenClassification.from_pretrained(str(Path(models_path) / f))
        .eval()
        .cuda()
        for f in PUNCT_MODELS
    ]
    tokenizer = AutoTokenizer.from_pretrained(PUNCT_MODELS[0])

    # post process
    for i, text in enumerate(texts):
        pred = text["text"].strip()
        pred = fix_repetition(pred, max_count=8)
        pred = punctuate(pred, tokenizer, models)
        if pred[-1] not in ["।", "?", ","]:
            pred = pred + "।"
        results.append((dia_results[i][0], pred))

    return results
