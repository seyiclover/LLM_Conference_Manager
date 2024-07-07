import numpy as np
import io, os
import torch
import soundfile as sf
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer
import subprocess
import logging

# .env 파일 로드
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Hugging Face token
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# 모델 로드
model = AutoModelForSpeechSeq2Seq.from_pretrained("NexoChatFuture/whisper-small-youtube-extra", use_auth_token=HUGGINGFACE_TOKEN)
processor = AutoProcessor.from_pretrained("NexoChatFuture/whisper-small-youtube-extra", use_auth_token=HUGGINGFACE_TOKEN)
tokenizer = AutoTokenizer.from_pretrained("NexoChatFuture/whisper-small-youtube-extra", use_auth_token=HUGGINGFACE_TOKEN)
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=HUGGINGFACE_TOKEN)

# 모델을 GPU로 이동 (가능한 경우)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

SR = 16000
CHUNK_DURATION = 30  # seconds

# 로깅 설정
logging.basicConfig(level=logging.ERROR)

def load_audio(path):
    """오디오 파일을 로드하고 변환"""
    try:
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
        return audio
    except Exception as e:
        logging.error(f"오류 발생: {e}")
        raise

def return_transcription(audio_data):
    """오디오 데이터를 전사하여 텍스트 반환"""
    try:
        audio = np.frombuffer(audio_data.read(), dtype=np.float32)
        total_duration = len(audio) / SR
        transcriptions = []

        for start in range(0, int(total_duration), CHUNK_DURATION):
            end = min(start + CHUNK_DURATION, total_duration)
            chunk = audio[int(start * SR):int(end * SR)]
            inputs = processor(chunk, return_tensors="pt", sampling_rate=SR)
            inputs = inputs.to(device)
            with torch.no_grad():
                predicted_ids = model.generate(inputs.input_features)
            transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcriptions.append(transcription)

        return ' '.join(transcriptions)
    except Exception as e:
        logging.error(f"오류 발생: {e}")
        raise

def merge_speaker_segments(segments):
    """화자 세그먼트를 병합"""
    merged_segments = []
    current_speaker = None
    current_segment = None

    for segment in segments:
        speaker = segment['speaker']
        if speaker == current_speaker:
            current_segment['end'] = segment['end']
            current_segment['audio'] += segment['audio']
        else:
            if current_segment:
                merged_segments.append(current_segment)
            current_speaker = speaker
            current_segment = {
                'speaker': speaker,
                'start': segment['start'],
                'end': segment['end'],
                'audio': segment['audio'],
            }

    if current_segment:
        merged_segments.append(current_segment)

    return merged_segments

def rename_speakers(segments):
    """화자 이름을 고유하게 재명명"""
    unique_speakers = list({segment['speaker'] for segment in segments})
    speaker_map = {speaker: f'참여자 {i+1}' for i, speaker in enumerate(unique_speakers)}

    for segment in segments:
        segment['speaker'] = speaker_map[segment['speaker']]

    return segments

def speaker_diarize(path, num_speakers):
    try:
        audio = load_audio(path)
        diarization = pipeline(path, num_speakers=num_speakers)

        segments = []
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
            segments.append({
                'speaker': speaker,
                'start': turn.start,
                'end': turn.end,
                'audio': audio_data.read()
            })

        segments = merge_speaker_segments(segments)

        for segment in segments:
            audio_data = io.BytesIO(segment['audio'])
            segment['transcription'] = return_transcription(audio_data)

        segments = rename_speakers(segments)
        return segments
    except Exception as e:
        logging.error(f"오류 발생: {e}")
        raise
