<div align="center">

# JSON RVC Inference

</div>

### Information
JSON RVC Inference is the same [advanced version of RVC](https://github.com/ArkanDash/Advanced-RVC-Inference) with JSON file to select desired model to download and load on the inference.
Best use case for google colab enviroment.

Please support the original RVC. This inference won't be possible to make without it.<br />
[![Original RVC Repository](https://img.shields.io/badge/Github-Original%20RVC%20Repository-blue?style=for-the-badge&logo=github)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

#### Features
- Support V1 & V2 Model ✅
- Model downloader using JSON file [Internet required for downloading voice model] ✅
- Youtube Audio Downloader ✅
- Voice Splitter [Internet required for downloading splitter model] ✅
- Microphone Support ✅
- TTS Support ✅

### Installation

1. Install Pytorch <br />
    - CPU only (any OS)
    ```bash
    pip install torch torchvision torchaudio
    ```
    - Nvidia (CUDA used)
    ```bash
    # For Windows (Due to flashv2 not supported in windows, Issue: https://github.com/Dao-AILab/flash-attention/issues/345#issuecomment-1747473481)
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    # Other (Linux, etc)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

2. Install Dependencies<br />
```bash
pip install -r requirements.txt
```
3. Install [ffmpeg](https://ffmpeg.org/)

4. Download Pre-model 
```bash
# Hubert Model
https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt
# Save it to /assets/hubert/hubert_base.pt

# RVMPE (rmvpe pitch extraction, Optional)
https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt
# Save it to /assets/rvmpe/rmvpe.pt
```

### Run WebUI <br />

For Windows:
```bash
Open run.bat
```
For Other:
```bash
python infer.py
```
