import os
import sys
now_dir = os.getcwd()
sys.path.append(now_dir)
from infer.modules.vc.modules import VC
from infer.modules.vc.utils import download_and_split_audio, combine_audio
from infer.lib.setting import change_audio_mode, show_description, use_microphone
from configs.config import Config
import numpy as np
import gradio as gr
import subprocess
import json
import shutil
import logging
import zipfile
import glob
import asyncio
import edge_tts

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
config = Config()
vc = VC(config)

tts_voice_list = asyncio.new_event_loop().run_until_complete(edge_tts.list_voices())
tts_voices = [f"{v['ShortName']}-{v['Gender']}" for v in tts_voice_list]

os.makedirs("models", exist_ok=True)
os.makedirs(os.path.join("models", "weights"), exist_ok=True)
os.makedirs(os.path.join("models", "indexs"), exist_ok=True)
os.makedirs(os.path.join("models", "covers"), exist_ok=True)

models_path = "models"
weights_path = os.path.join("models", "weights")
indexs_path = os.path.join("models", "indexs")
covers_path = os.path.join("models", "covers")

force_support = None
if config.force_support is False:
    if config.device == "mps" or config.device == "cpu":
        force_support = False
else:
    print("\033[93mWARNING: Unsupported feature is enabled.\033[0m")
    print("\033[93mWARNING: It may not work properly.\033[0m")
    force_support = True

audio_mode = []
f0method_mode = []
f0method_info = ""

if force_support is False:
    audio_mode = ["Upload audio", "Input path", "TTS Audio"]
    f0method_mode = ["pm", "rmvpe", "harvest"]
else:
    audio_mode = ["Upload audio", "Input path", "Youtube", "TTS Audio"]
    f0method_mode = ["pm", "rmvpe", "harvest", "crepe"]

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
    models = data["model_data"]
    for item in models:
        if character == item[1]:
            cover_filename = os.path.splitext(item[0])[0] + os.path.splitext(item[2])[1]
            log1 = subprocess.run(['wget', '-P', temp_path, item[2]])
            log2 = subprocess.run(['wget', '-O', os.path.join(covers_path, cover_filename), item[3]])
    zip_files = glob.glob(os.path.join(temp_path, '*.zip'))
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_path)
    for root, dirs, files in os.walk(temp_path):
        for file in files:
            if file.endswith(".pth"):
                weight_file = f"{character}.{file.split('.')[1]}"
                os.rename(os.path.join(root, file), os.path.join(root, weight_file))
                shutil.move(os.path.join(root, weight_file), os.path.join(weights_path, weight_file))
            elif file.endswith(".index"):
                if character not in file:
                    index_file = f"{file.split('.')[0]}_{character}.{file.split('.')[1]}"
                    os.rename(os.path.join(root, file), os.path.join(root, index_file))
                    shutil.move(os.path.join(root, index_file), os.path.join(indexs_path, index_file))
                else:
                    shutil.move(os.path.join(root, file), os.path.join(indexs_path, file))
        for dir in dirs:
            for root_dirs, _, files_dirs in os.walk(os.path.join(root, dir)):
                for file_dir in files_dirs:
                    if file_dir.endswith(".pth"):
                        weight_file = f"{character}.{file_dir.split('.')[1]}"
                        os.rename(os.path.join(root_dirs, file_dir), os.path.join(root_dirs, weight_file))
                        shutil.move(os.path.join(root_dirs, weight_file), os.path.join(weights_path, weight_file))
                    elif file_dir.endswith(".index"):
                        if character not in file_dir:
                            index_file = f"{file_dir.split('.')[0]}_{character}.{file_dir.split('.')[1]}"
                            os.rename(os.path.join(root_dirs, file_dir), os.path.join(root_dirs, index_file))
                            shutil.move(os.path.join(root_dirs, index_file), os.path.join(indexs_path, index_file))
                        else:
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

with gr.Blocks(title="RVC WebUI", theme=gr.themes.Base()) as app:
    gr.Markdown("<center> # RVC WebUI")
    gr.Markdown("v1.0.0Beta")
    with gr.Tabs():
        with gr.TabItem("Inference"):
            with gr.Row():
                modelSelect = gr.Dropdown(label="Model", choices=sorted(modelList))
                downloadModel = gr.Button("Download Model", variant="primary")
                downloadModel.click(fn=download_model, inputs=[modelSelect], outputs=[])
            with gr.Row():
                sid0 = gr.Dropdown(label="Selected Model", choices=sorted(names))
                file_index = gr.Dropdown(
                    label="Index file dropdown",
                    choices=sorted(indexs),
                    interactive=True,
                )
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label="Speaker ID",
                    value=0,
                    visible=True,
                    interactive=True,
                )
                with gr.Column():
                    refresh_button = gr.Button("Refresh model", variant="primary")
                    clean_button = gr.Button("Clean memory", variant="primary")
                clean_button.click(fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean")
            with gr.TabItem("Inference Setting"):
                with gr.Row():
                    with gr.Column():
                        vc_audio_mode = gr.Dropdown(label="Input voice", choices=audio_mode, value="Upload audio", visible=True, interactive=True)
                        # Upload Audio
                        vc_upload = gr.Audio(label="Upload audio file", sources="upload", visible=True, interactive=True)
                        vc_microphone_mode = gr.Checkbox(label="Use Microphone", value=False, visible=True, interactive=True)
                        # Audio Path
                        vc_audio_input = gr.Textbox(label="Audio Input Path", placeholder="C:\\Users\\Desktop\\audio_example.wav", visible=False, interactive=True)
                        # Youtube Audio
                        vc_link = gr.Textbox(label="Youtube URL", visible=False, placeholder="https://www.youtube.com/watch?v=...", interactive=True)
                        vc_split_model = gr.Dropdown(label="Splitter Model", choices=["hdemucs_mmi", "htdemucs", "htdemucs_ft", "mdx", "mdx_extra", "mdx_extra_q"], allow_custom_value=False, visible=False, value="htdemucs", interactive=True)
                        vc_download_button = gr.Button("Download Audio", variant="primary", visible=False)
                        vc_vocal_preview = gr.Audio(label="Vocal Preview", visible=False)
                        # TTS Audio
                        vc_tts_text = gr.Textbox(label="TTS text", placeholder="Hello world", visible=False, interactive=True)
                        vc_tts_voice = gr.Dropdown(label="Edge-tts speaker", choices=tts_voices, visible=False, allow_custom_value=False, value="en-US-AnaNeural-Female", interactive=True)
                    with gr.Column():
                        f0method0 = gr.Radio(
                            label="Pitch extraction algorithm",
                            choices=f0method_mode
                            if config.dml == False
                            else ["pm", "harvest", "rmvpe"],
                            value="rmvpe",
                            interactive=True,
                        )
                        vc_transform0 = gr.Slider(
                            label="Transpose",
                            minimum=-256,
                            maximum=256,
                            step=0.01,
                            value=0,
                            interactive=True,
                        )
                        index_rate = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Retrieval feature ratio",
                            value=0.75,
                            interactive=True,
                        )
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
                        refresh_button.click(
                            fn=change_choices,
                            inputs=[],
                            outputs=[sid0, file_index],
                            api_name="infer_refresh",
                        )
                    with gr.Column():
                        but0 = gr.Button("Run", variant="primary")
                        vc_output = gr.Audio(label="Output Audio")
                        vc_combined = gr.Button("Combine", variant="primary")
                        vc_combined_output = gr.Audio(label="Combined Audio")
        with gr.TabItem("Log"):
            gr.Markdown("## Log")
            vc_log = gr.Textbox(label="Output Log")
        with gr.TabItem("Settings"):
            gr.Markdown("## Setting")
            description_mode = gr.Checkbox(label="Show description", value=False)
            description_mode.change(
                fn=show_description,
                inputs=description_mode,
                outputs=[
                    vc_audio_input,
                    vc_link,
                    vc_split_model,
                    vc_tts_text,
                    vc_tts_voice,
                    f0method0,
                    vc_transform0,
                    index_rate,
                    resample_sr0,
                    rms_mix_rate0,
                    protect0,
                    filter_radius0
                ]
            )
        but0.click(
            vc.vc_single,
            [
                spk_item,
                vc_audio_mode,
                vc_audio_input,
                vc_upload,
                vc_tts_text,
                vc_tts_voice,
                vc_transform0,
                f0method0,
                file_index,
                index_rate,
                filter_radius0,
                resample_sr0,
                rms_mix_rate0,
                protect0,
            ],
            [vc_log, vc_output],
            api_name="infer_convert",
        )
        sid0.change(
            fn=vc.get_vc,
            inputs=[sid0, protect0],
            outputs=[spk_item, protect0, file_index],
            api_name="infer_change_voice",
        )
        vc_microphone_mode.change(
            fn=use_microphone,
            inputs=vc_microphone_mode,
            outputs=vc_upload
        )
        vc_download_button.click(
            fn=download_and_split_audio,
            inputs=[vc_link, vc_split_model],
            outputs=[vc_vocal_preview, vc_log]
        )
        vc_combined.click(
            fn=combine_audio,
            inputs=[vc_split_model],
            outputs=[vc_combined_output, vc_log]
        )
        vc_audio_mode.change(
            fn=change_audio_mode,
            inputs=[vc_audio_mode],
            outputs=[
                vc_upload,
                vc_microphone_mode,
                vc_audio_input,
                vc_link,
                vc_split_model,
                vc_download_button,
                vc_vocal_preview,
                vc_tts_text,
                vc_tts_voice,
                vc_combined,
                vc_combined_output,
            ]
        )
    if config.iscolab:
        app.queue(max_size=1022).launch(share=True)
    else:
        app.queue(max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )
