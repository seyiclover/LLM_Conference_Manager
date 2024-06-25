import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer
import torch
import torchaudio
from fastapi.responses import HTMLResponse
import whisper

# Hugging Face 토큰
HUGGINGFACE_TOKEN = "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"

app = FastAPI()

# 모델, 프로세서, 토크나이저를 각각의 저장소에서 로드
model = AutoModelForSpeechSeq2Seq.from_pretrained("svenskpotatis/sample-project-stt-meeting-model-deVad", token=HUGGINGFACE_TOKEN)
processor = AutoProcessor.from_pretrained("svenskpotatis/sample-project-stt-model-meeting-processor-deVad", token=HUGGINGFACE_TOKEN)
tokenizer = AutoTokenizer.from_pretrained("svenskpotatis/sample-project-stt-model-meeting-tokenizer-deVad", token=HUGGINGFACE_TOKEN)


@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <html>
        <head>
            <title>Audio to Text</title>
        </head>
        <body>
            <h1>Upload audio file</h1>
            <form action="/audioToText" method="post" enctype="multipart/form-data">
                Title: <input type="text" name="title"><br>
                Audio File: <input type="file" name="audio"><br>
                <input type="submit" value="Submit">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/audioToText")
async def audio_to_text(audio: UploadFile = File(...)):
    try:
        # 오디오 파일을 임시로 저장
        audio_bytes = await audio.read()
        audio_path = "temp_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        # 오디오 로딩 및 전처리
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        # 모델 입력 준비
        inputs = processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt")

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

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9090)
