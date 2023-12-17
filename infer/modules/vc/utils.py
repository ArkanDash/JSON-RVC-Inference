import os
import ffmpeg
import subprocess
import random

from fairseq import checkpoint_utils
from pytube import YouTube

def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )

def load_hubert(config):
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["assets/hubert/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()

async def download_and_split_audio(url, model):
    audio = YouTube(url).streams.get_audio_only()
    audio.download(filename="audio.wav", output_path="output/yt",skip_existing=False)
    command = f"demucs --two-stems=vocals -n {model} output/yt/audio.wav -o output"
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    vocal = f"output/{model}/audio/vocals.wav"
    inst = f"output/{model}/audio/no_vocals.wav"
    return vocal, result

async def combine_audio(model):
    vocal = f"output/{model}/audio/vocals.wav"
    inst = f"output/{model}/audio/no_vocals.wav"
    os.mkdir(os.path.join("output", "combined"))
    random_number = random.randint(0, 1000000)
    output_path = os.path.join("output", "combined", f"combined_{random_number}.wav")
    while output_path in os.listdir(os.path.join("output", "combined")):
        random_number = random.randint(0, 1000000)
        output_path = os.path.join("output", "combined", f"combined_{random_number}.wav")
    command = f"ffmpeg -i {inst} -i {vocal} -filter_complex amix=inputs=2:duration=longest -vn {output_path}"
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    return output_path, result
