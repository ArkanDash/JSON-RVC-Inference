import os
import sys
from dotenv import load_dotenv

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()
from infer.modules.vc.modules import VC
from configs.config import Config
import numpy as np
import gradio as gr
import subprocess
import json
import shutil
import logging
import zipfile
import glob

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
config = Config()
vc = VC(config)

os.makedirs("models", exist_ok=True)
os.makedirs(os.path.join("models", "weights"), exist_ok=True)
os.makedirs(os.path.join("models", "indexs"), exist_ok=True)
os.makedirs(os.path.join("models", "covers"), exist_ok=True)

models_path = "models"
weights_path = os.path.join("models", "weights")
indexs_path = os.path.join("models", "indexs")
covers_path = os.path.join("models", "covers")

json_files = []
for root, dirs, files in os.walk(models_path):
    for file in files:
        if file.endswith(".json"):
            json_files.append(os.path.join(root, file))
with open(json_files[0], 'r') as file:
    data = json.load(file)
modelList = data["list"]
names = []
indexs = []
for name in os.listdir(weights_path):
    if name.endswith(".pth"):
        names.append(name)
for name in os.listdir(indexs_path):
    if name.endswith(".index"):
        indexs.append(name)

def download_model(character):
    os.makedirs("TEMP", exist_ok=True)
    temp_path = "TEMP"
    if "v2" in character:
        ver_2 = data["version_2"]
        for item in ver_2:
            if character in item[0]:
                cover_filename = os.path.splitext(item[0])[0] + os.path.splitext(item[2])[1]
                subprocess.run(['wget', '-P', temp_path, item[1]])
                subprocess.run(['wget', '-O', os.path.join(covers_path, cover_filename), item[2]])
    else:
        ver_1 = data["version_1"]
        for item in ver_1:
            if character in item[0]:
                cover_filename = os.path.splitext(item[0])[0] + os.path.splitext(item[2])[1]
                subprocess.run(['wget', '-P', temp_path, item[1]])
                subprocess.run(['wget', '-O', os.path.join(covers_path, cover_filename), item[2]])
    zip_files = glob.glob(os.path.join(temp_path, '*.zip'))
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_path)
    for root, dirs, files in os.walk(temp_path):
        for file in files:
            model_name = file.split(".")[0]
            if file.endswith(".pth"):
                shutil.move(os.path.join(root, file), os.path.join(weights_path, file))
            elif file.endswith(".index"):
                if model_name not in index_name:
                    new_index_name = f"{root}/{file.split('.')[0]}_{model_name}.{file.split('.')[1]}"
                    os.rename(os.path.join(root, file), index_name)
                shutil.move(os.path.join(root, file), os.path.join(indexs_path, file))
        for dir in dirs:
            for root_dirs, _, files_dirs in os.walk(os.path.join(root, dir)):
                for file_dir in files_dirs:
                    model_name = file.split(".")[0]
                    if file_dir.endswith(".pth"):
                        shutil.move(os.path.join(root_dirs, file_dir), os.path.join(weights_path, file_dir))
                    elif file_dir.endswith(".index"):
                        if model_name not in index_name:
                            new_index_name = f"{root_dirs}/{file_dir.split('.')[0]}_{model_name}.{file_dir.split('.')[1]}"
                            os.rename(os.path.join(root_dirs, file_dir), index_name)
                        shutil.move(os.path.join(root_dirs, file_dir), os.path.join(indexs_path, file_dir))
    shutil.rmtree(temp_path)

def change_choices():
    names = []
    indexs = []
    for name in os.listdir(weights_path):
        if name.endswith(".pth"):
            names.append(name)
    for name in os.listdir(indexs_path):
        if name.endswith(".index"):
            indexs.append(name)
    return { "choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(indexs),
        "__type__": "update",
    }

def clean():
    return {"value": "", "__type__": "update"}

with gr.Blocks(title="RVC WebUI") as app:
    gr.Markdown("## RVC WebUI")
    gr.Markdown(
        value="Alpha testing."
    )
    with gr.Tabs():
        with gr.TabItem("Inference"):
            with gr.Row():
                modelSelect = gr.Dropdown(label="Model", choices=sorted(modelList))
                downloadModel = gr.Button("Download Model", variant="primary")
                downloadModel.click(fn=download_model, inputs=[modelSelect], outputs=[])
            with gr.Row():
                sid0 = gr.Dropdown(label="Selected Model", choices=sorted(names))
                with gr.Column():
                    refresh_button = gr.Button("Refresh model", variant="primary")
                    clean_button = gr.Button("Clean memory", variant="primary")
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label="Speaker ID",
                    value=0,
                    visible=True,
                    interactive=True,
                )
                clean_button.click(
                    fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean"
                )
            with gr.TabItem("Inference Setting"):
                with gr.Row():
                    with gr.Column():
                        vc_transform0 = gr.Number(
                            label="Transpose", value=0
                        )
                        input_audio0 = gr.Textbox(
                            label="Audio Input Path",
                            placeholder="C:\\Users\\Desktop\\audio_example.wav",
                        )
                        file_index = gr.Dropdown(
                            label="Index file dropdown",
                            choices=sorted(indexs),
                            interactive=True,
                        )
                        f0method0 = gr.Radio(
                            label="Pitch extraction algorithm",
                            choices=["pm", "harvest", "crepe", "rmvpe"]
                            if config.dml == False
                            else ["pm", "harvest", "rmvpe"],
                            value="rmvpe",
                            interactive=True,
                        )
                    with gr.Column():
                        resample_sr0 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label="Resample the output audio",
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate0 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Volume Envelope",
                            value=0.25,
                            interactive=True,
                        )
                        protect0 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label="Voice Protection",
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                        filter_radius0 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label="Apply Median Filtering",
                            value=3,
                            step=1,
                            interactive=True,
                        )
                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Retrieval feature ratio",
                            value=0.75,
                            interactive=True,
                        )
                        f0_file = gr.File(
                            label="F0 curve file (Optional)",
                            visible=False,
                        )
                        refresh_button.click(
                            fn=change_choices,
                            inputs=[],
                            outputs=[sid0, file_index],
                            api_name="infer_refresh",
                        )
                    with gr.Column():
                        but0 = gr.Button("Run", variant="primary")
                        vc_output1 = gr.Textbox(label="Output Log")
                        vc_output2 = gr.Audio(label="Output Audio")
                        but0.click(
                            vc.vc_single,
                            [
                                spk_item,
                                input_audio0,
                                vc_transform0,
                                f0_file,
                                f0method0,
                                file_index,
                                index_rate1,
                                filter_radius0,
                                resample_sr0,
                                rms_mix_rate0,
                                protect0,
                            ],
                            [vc_output1, vc_output2],
                            api_name="infer_convert",
                        )
                sid0.change(
                    fn=vc.get_vc,
                    inputs=[sid0, protect0],
                    outputs=[spk_item, protect0, file_index],
                    api_name="infer_change_voice",
                )
    if config.iscolab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )
