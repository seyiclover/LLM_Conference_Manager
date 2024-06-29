# uvicorn==0.30.1
# fastapi==0.111.0
# transformers==4.42.3

# pytorch==2.3.0
# pytorch-lightning==2.3.0
# torch-audiomentations==0.11.1
# torchaudio==2.3.0
# torchvision==0.18.0

# numpy==1.24.3
# soundfile==0.12.1
# pyannote-audio==3.3.1


import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer
import torch
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf
from pyannote.audio import Pipeline
from subprocess import run

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 키 및 토큰 설정 
HUGGINGFACE_TOKEN = ""
HYPERCLOVA_API_KEY = ''
HYPERCLOVA_API_KEY_PRIMARY_VAL = ''
HYPERCLOVA_REQUEST_ID = ''
HYPERCLOVA_HOST = 'https://clovastudio.stream.ntruss.com'

# 모델 및 프로세서 로드
model = AutoModelForSpeechSeq2Seq.from_pretrained("svenskpotatis/sample-project-stt-meeting-model-deVad", use_auth_token=HUGGINGFACE_TOKEN)
processor = AutoProcessor.from_pretrained("svenskpotatis/sample-project-stt-model-meeting-processor-deVad", use_auth_token=HUGGINGFACE_TOKEN)
tokenizer = AutoTokenizer.from_pretrained("svenskpotatis/sample-project-stt-model-meeting-tokenizer-deVad", use_auth_token=HUGGINGFACE_TOKEN)
diarization_pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=HUGGINGFACE_TOKEN)

# 오디오 파일을 로드하고 처리하는 함수
def load_audio(path):
    output_path = './output_audio.wav'
    cmd = ["ffmpeg", "-nostdin",
           "-threads", "0",
           "-i", path,
           "-f", "s16le",
           "-ac", "1",
           "-acodec", "pcm_s16le",
           "-ar", "16000",
           "-"]
    out = run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    sf.write(output_path, audio, 16000)
    return audio, output_path

# 화자분리, stt 전사
@app.post("/extract_text")
async def extract_text(media: UploadFile = File(...), num_speakers: int = Form(...)):
    temp_input_path = None

    try:
        content = await media.read()
        file_extension = os.path.splitext(media.filename)[1]
        temp_input_path = f"temp_input{file_extension}"

        with open(temp_input_path, "wb") as f:
            f.write(content)

        # 오디오/비디오 전처리
        audio, audio_path = load_audio(temp_input_path)
        print("전처리 완료")

        # 화자 분리 
        diarization = diarization_pipeline({'audio': audio_path}, num_speakers=num_speakers)
        print("화자분리 완료")
        
        transcriptions = [] # 배열로 반환할 경우
        # transcriptions = '' # 문자열로 반환할 경우

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_audio = audio[int(turn.start * 16000):int(turn.end * 16000)]
            inputs = processor(speaker_audio, sampling_rate=16000, return_tensors="pt")

            with torch.no_grad():
                predicted_ids = model.generate(inputs.input_features, num_beams=5)

            transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcriptions.append(f"{speaker}: {transcription}") # 문자열
            # transcriptions += f"{speaker}: {transcription}" # 배열

        print("전사 완료")

        return {"transcriptions": transcriptions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000) 
