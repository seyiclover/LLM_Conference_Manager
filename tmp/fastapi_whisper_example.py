# ffmpeg version 7.0.1 Copyright (c) 2000-2024 the FFmpeg developers
# uvicorn.__version__==0.30.1
# fastapi.__version__==0.111.0
# transformers.__version__==4.41.2
# torch.__version__==2.3.1
# whisper.__version__==20231117
# numpy.__version__==1.24.3
# soundfile.__version__==0.12.1


import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer
import torch
from fastapi.responses import HTMLResponse
import numpy as np
from subprocess import run, CalledProcessError
import os

# Hugging Face 토큰
HUGGINGFACE_TOKEN = "huggingface_token"

app = FastAPI()

# 모델, 프로세서, 토크나이저를 각각의 저장소에서 로드
model = AutoModelForSpeechSeq2Seq.from_pretrained("NexoChatFuture/whisper-small-youtube", token=HUGGINGFACE_TOKEN)
processor = AutoProcessor.from_pretrained("NexoChatFuture/whisper-small-youtube-tokenizer", token=HUGGINGFACE_TOKEN)
tokenizer = AutoTokenizer.from_pretrained("NexoChatFuture/whisper-small-youtube-processor", token=HUGGINGFACE_TOKEN)


@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <html>
        <head>
            <title>Audio to Text</title>
        </head>
        <body>
            <h1>Upload media file</h1>
            <form action="/audioToText" method="post" enctype="multipart/form-data">
                Title: <input type="text" name="title"><br>
                Media File: <input type="file" name="media"><br>
                <input type="submit" value="Submit">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/audioToText")
async def audio_to_text(media: UploadFile = File(...)):

    temp_input_path = None
    
    try:
        # 오디오 파일을 임시로 저장
        content = await media.read()
        file_extension = os.path.splitext(media.filename)[1]
        temp_input_path = f"temp_input{file_extension}"
        
        with open(temp_input_path, "wb") as f:
            f.write(content)

        # FFmpeg 명령어 설정
        # Whisper input 형태의 오디오로 전처리
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", temp_input_path,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-"
        ]

        try:
            out = run(cmd, capture_output=True, check=True).stdout
        except CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
            

        audio =  np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

        # 모델 입력 준비
        inputs = processor(audio, return_tensors="pt")

        # 모델을 사용한 예측
        with torch.no_grad():
            predicted_ids = model.generate(inputs.input_features, num_beams=5)
            
        # 예측된 토큰을 텍스트로 디코딩
        transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print("Transcription:", transcription)

        return {
            'text': transcription
        }
    except Exception as e:
        print(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 임시 파일 삭제
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9090)
