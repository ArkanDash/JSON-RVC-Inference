import gradio as gr

def change_audio_mode(vc_audio_mode):
    if vc_audio_mode == "Upload audio":
        return (
            # Upload Audio
            gr.Audio(visible=True),
            # Audio Path
            gr.Textbox(visible=False),
            # Youtube Audio
            gr.Textbox(visible=False),
            gr.Dropdown(visible=False),
            gr.Button(visible=False),
            gr.Audio(visible=False),
            # TTS Audio
            gr.Textbox(visible=False),
            gr.Dropdown(visible=False),
            # Combine Audio
            gr.Button(visible=False),
            gr.Audio(visible=False),
        )
    elif vc_audio_mode == "Input path":
        return (
            # Upload Audio
            gr.Audio(visible=False),
            # Audio Path
            gr.Textbox(visible=True),
            # Youtube Audio
            gr.Textbox(visible=False),
            gr.Dropdown(visible=False),
            gr.Button(visible=False),
            gr.Audio(visible=False),
            # TTS Audio
            gr.Textbox(visible=False),
            gr.Dropdown(visible=False),
            # Combine Audio
            gr.Button(visible=False),
            gr.Audio(visible=False),
        )
    elif vc_audio_mode == "Youtube":
        return (
            # Upload Audio
            gr.Audio(visible=False),
            # Audio Path
            gr.Textbox(visible=False),
            # Youtube Audio
            gr.Textbox(visible=True),
            gr.Dropdown(visible=True),
            gr.Button(visible=True),
            gr.Audio(visible=True),
            # TTS Audio
            gr.Textbox(visible=False),
            gr.Dropdown(visible=False),
            # Combine Audio
            gr.Button(visible=True),
            gr.Audio(visible=True),
        )
    elif vc_audio_mode == "TTS Audio":
        return (
            # Upload Audio
            gr.Audio(visible=False),
            # Audio Path
            gr.Textbox(visible=False),
            # Youtube Audio
            gr.Textbox(visible=False),
            gr.Dropdown(visible=False),
            gr.Button(visible=False),
            gr.Audio(visible=False),
            # TTS Audio
            gr.Textbox(visible=True),
            gr.Dropdown(visible=True),
            # Combine Audio
            gr.Button(visible=False),
            gr.Audio(visible=False),
        )

def show_description(isTrue):
    if isTrue is True:
        return(
            # Audio Path
            gr.Textbox(info="Select the audio file (Example: C:\\Users\\Desktop\\audio_example.wav)"),
            # Youtube Audio
            gr.Textbox(info="Input your youtube link (Example: https://www.youtube.com/watch?v=Nc0sB1Bmf-A"),
            gr.Dropdown(info="Select the splitter model (Default: htdemucs)"),
            # TTS Audio
            gr.Textbox(info="Text to speech input"),
            gr.Dropdown(info="List of TTS speaker (Default: en-US-AnaNeural-Female)"),
            # Converter Settings
            gr.Radio(info="PM is fast, Harvest is good but extremely slow, Rvmpe is alternative to harvest (might be better), and Crepe effect is good but requires GPU (Default: PM)"),
            gr.Slider(info="Number of semitone to shift pitch (Use 0 for no effect)"),
            gr.Slider(info="Ratio between original voice and model voice (Default: 0.6)"),
            gr.Slider(info="Resample the output audio in post-processing to the final sample rate. (Set to 0 for no resampling)"),
            gr.Slider(info="Use the volume envelope of the input to replace or mix with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is used."),
            gr.Slider(info="Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Decrease the value to increase protection, but it may reduce indexing accuracy. (Set to 0.5 to disable)"),
            gr.Slider(info="The value represents the filter radius and can reduce breathiness."),
        )
    else:
        return(
            # Audio Path
            gr.Textbox(info=""),
            # Youtube Audio
            gr.Textbox(info=""),
            gr.Dropdown(info=""),
            # TTS Audio
            gr.Textbox(info=""),
            gr.Dropdown(info=""),
            # Converter Settings
            gr.Radio(info=""),
            gr.Slider(info=""),
            gr.Slider(info=""),
            gr.Slider(info=""),
            gr.Slider(info=""),
            gr.Slider(info=""),
            gr.Slider(info=""),
        )