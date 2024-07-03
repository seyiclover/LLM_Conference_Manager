from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
import numpy as np
import pandas as pd
import io
import os
import torch
import soundfile as sf
from pyannote.audio import Pipeline
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer
import subprocess

app = FastAPI()

# Hugging Face token
HUGGINGFACE_TOKEN = "HUGGINGFACE_TOKEN"

# model load
model = AutoModelForSpeechSeq2Seq.from_pretrained("NexoChatFuture/whisper-small-youtube-extra", token=HUGGINGFACE_TOKEN)
processor = AutoProcessor.from_pretrained("NexoChatFuture/whisper-small-youtube-extra", token=HUGGINGFACE_TOKEN)
tokenizer = AutoTokenizer.from_pretrained("NexoChatFuture/whisper-small-youtube-extra", token=HUGGINGFACE_TOKEN)
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=HUGGINGFACE_TOKEN)

# move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
pipeline.to(device)

SR = 16000
CHUNK_DURATION = 30  # seconds

@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <html>
        <head>
            <title>Audio to Text</title>
        </head>
        <body>
            <h1>Upload media file</h1>
            <form action="/diarize-and-transcribe/" method="post" enctype="multipart/form-data">
                Title: <input type="text" name="title"><br>
                Media File: <input type="file" name="media"><br>
                Number of Speakers: <input type="number" name="num_speakers" value="1"><br>
                <input type="submit" value="Submit">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/diarize-and-transcribe/")
async def diarize_and_transcribe(media: UploadFile = File(...), num_speakers: int = Form(...)):
    temp_input_path = None

    try:
        content = await media.read()
        file_extension = os.path.splitext(media.filename)[1]
        temp_input_path = f"temp_input{file_extension}"
        
        with open(temp_input_path, "wb") as f:
            f.write(content)

        result_df = speaker_diarize(temp_input_path, num_speakers)

        for index, row in enumerate(result_df.itertuples()):
            print(f'{row.speakers}: {row.transcription}')

        result = result_df.to_dict(orient='records')
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)

def load_audio(path):
    output_path = './output_audio.wav'
    cmd = [
        "ffmpeg", "-nostdin",
        "-threads", "0",
        "-i", path,
        "-f", "f32le",
        "-ac", "1",
        "-acodec", "pcm_f32le",
        "-ar", str(SR),
        "-"
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(out, np.float32)
    sf.write(output_path, audio, SR)
    return audio, output_path

def return_transcription(audio_data):
    audio = np.frombuffer(audio_data.read(), dtype=np.float32)
    total_duration = len(audio) / SR
    transcriptions = []

    for start in range(0, int(total_duration), CHUNK_DURATION):
        end = min(start + CHUNK_DURATION, total_duration)
        chunk = audio[int(start * SR):int(end * SR)]
        inputs = processor(chunk, return_tensors="pt", sampling_rate=SR)
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        with torch.no_grad():
            predicted_ids = model.generate(inputs.input_features)
        transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        transcriptions.append(transcription)

    return ' '.join(transcriptions)

def merge_speaker_rows(df):
    merged_data = []
    current_speaker = None
    current_start = None
    current_end = None
    current_audio = []
    current_transcription = []

    for index, row in df.iterrows():
        if row['speakers'] == current_speaker:
            current_end = row['end_timestamp']
            current_audio.append(row['audio'])
            current_transcription.append(row.get('transcription', ''))
        else:
            if current_speaker is not None:
                merged_data.append({
                    'speakers': current_speaker,
                    'start_timestamp': current_start,
                    'end_timestamp': current_end,
                    'audio': b''.join(current_audio),
                    'transcription': ' '.join(current_transcription).strip()
                })
            current_speaker = row['speakers']
            current_start = row['start_timestamp']
            current_end = row['end_timestamp']
            current_audio = [row['audio']]
            current_transcription = [row.get('transcription', '')]

    if current_speaker is not None:
        merged_data.append({
            'speakers': current_speaker,
            'start_timestamp': current_start,
            'end_timestamp': current_end,
            'audio': b''.join(current_audio),
            'transcription': ' '.join(current_transcription).strip()
        })

    return pd.DataFrame(merged_data)

def rename_speakers(df):
    unique_speakers = df['speakers'].unique()
    speaker_map = {speaker: f'참여자 {i+1}' for i, speaker in enumerate(unique_speakers)}
    df['speakers'] = df['speakers'].map(speaker_map)
    return df

def speaker_diarize(path, num_speakers):
    audio, audio_path = load_audio(path)
    speakers = []
    start_timestamp = []
    end_timestamp = []
    audio_segments = []

    diarization = pipeline({'uri': 'unique_audio_identifier', 'audio': audio_path}, num_speakers=num_speakers)

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        cmd = [
            "ffmpeg", "-nostdin",
            "-i", path,
            "-ss", str(turn.start),
            "-to", str(turn.end),
            "-f", "f32le",
            "-ac", "1",
            "-acodec", "pcm_f32le",
            "-ar", str(SR),
            "pipe:1"
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise Exception(f"Error occurred during ffmpeg process: {stderr.decode()}")

        audio_data = io.BytesIO(stdout)

        speakers.append(speaker)
        start_timestamp.append(turn.start)
        end_timestamp.append(turn.end)
        audio_segments.append(audio_data.read())

    df = pd.DataFrame({
        'speakers': speakers,
        'start_timestamp': start_timestamp,
        'end_timestamp': end_timestamp,
        'audio': audio_segments
    })

    merged_df = merge_speaker_rows(df)

    transcriptions = []
    for _, row in merged_df.iterrows():
        audio_data = io.BytesIO(row['audio'])
        transcription = return_transcription(audio_data)
        transcriptions.append(transcription)

    merged_df['transcription'] = transcriptions

    merged_df.dropna(subset=['transcription'], inplace=True)

    final_merged_df = merge_speaker_rows(merged_df)
    final_output_df = final_merged_df.drop(columns=['audio', 'start_timestamp', 'end_timestamp'])
    final_output_df = final_output_df[final_output_df['transcription'] != '']
    final_output_df.reset_index(drop=True, inplace=True)
    final_output_df = rename_speakers(final_output_df)

    return final_output_df

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9090)
