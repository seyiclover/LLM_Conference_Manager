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
from fastapi.responses import StreamingResponse
import numpy as np
from subprocess import run, CalledProcessError
import os
import requests
import json
from fastapi.middleware.cors import CORSMiddleware

# Hugging Face 토큰
HUGGINGFACE_TOKEN = "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"

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델, 프로세서, 토크나이저를 각각의 저장소에서 로드
model = AutoModelForSpeechSeq2Seq.from_pretrained("svenskpotatis/sample-project-stt-meeting-model-deVad", token=HUGGINGFACE_TOKEN)
processor = AutoProcessor.from_pretrained("svenskpotatis/sample-project-stt-model-meeting-processor-deVad", token=HUGGINGFACE_TOKEN)
tokenizer = AutoTokenizer.from_pretrained("svenskpotatis/sample-project-stt-model-meeting-tokenizer-deVad", token=HUGGINGFACE_TOKEN)


# 하이퍼클로바 api 답변 생성
class CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self.host = host
        self.api_key = api_key
        self.api_key_primary_val = api_key_primary_val
        self.request_id = request_id

    async def execute(self, completion_request):
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self.api_key,
            'X-NCP-APIGW-API-KEY': self.api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self.request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }
        response = requests.post(self.host + '/testapp/v1/chat-completions/HCX-003',
                                 headers=headers, json=completion_request, stream=True)

        # 응답 스트리밍
        async def response_generator():
            try:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data:'):
                            # print(decoded_line)
                            data = json.loads(decoded_line[5:])

                            if data.get('stopReason') == 'stop_before':
                                break
                            content = data.get('message', {}).get('content')

                            if content is not None:
                                yield f"data:{content}\n\n"

            except Exception as e:
                yield f"data:Error:{str(e)}\n\n"
            finally:
                yield "data:[DONE]\n\n"

        return StreamingResponse(response_generator(), media_type="text/event-stream")

# STT 전사
@app.post("/audioToText")
async def audio_to_text(media: UploadFile = File(...), title: str = Form(...)):
    temp_input_path = None
    try:
        content = await media.read()
        file_extension = os.path.splitext(media.filename)[1]
        temp_input_path = f"temp_input{file_extension}"

        with open(temp_input_path, "wb") as f:
            f.write(content)

        cmd = ["ffmpeg", "-nostdin",
               "-threads", "0",
               "-i", temp_input_path,
               "-f", "s16le",
               "-ac", "1",
               "-acodec", "pcm_s16le",
               "-ar", "16000",
               "-"]

        try:
            out = run(cmd, capture_output=True, check=True).stdout

        except CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"Failed to process audio: {e.stderr.decode()}")

        audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
        inputs = processor(audio, return_tensors="pt")

        with torch.no_grad():
            predicted_ids = model.generate(inputs.input_features, num_beams=5)

        transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        return {"transcription": transcription}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)


# 하이퍼클로바 api 질문 답변 생성
@app.get("/answer_question")
async def answer_question(question: str, transcription: str):
    completion_executor = CompletionExecutor(
        host='https://clovastudio.stream.ntruss.com',
        api_key='NTA0MjU2MWZlZTcxNDJiY+nf9zcdD17iTceOgNKtLLKEfW4jEYHTw4MokkDqhRSd',
        api_key_primary_val='ejFEDwXImkcF0ZDgHkDxh6rv21AAzaL21zhgnN9h',
        request_id='ecc6593f-751a-4801-9900-b73d5a458d87'
    )

    preset_text = [
        {"role":"system","content":"사용자의 질문에 친절하게 답변해줘. ## 내용 ## 을 기반으로 설명해. 설명 중 ## 내용 ## 에 언급되지 않은 부분이 있다면, 언급되지 않은 부분이라고 반드시 명시하고 설명해.  "},
        {"role":"user","content": f"## 내용 ## \n{transcription}\n## 질문 ## \n{question}"}
    ]
    message = {
        'messages': preset_text,
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 2048,
        'temperature': 0.5,
        'repeatPenalty': 5.0,
        'stopBefore': [],
        'includeAiFilters': True,
        'seed': 0
    }
    return await completion_executor.execute(message)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9090)
