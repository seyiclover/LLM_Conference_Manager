import numpy as np
import io, os
import torch
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
model = AutoModelForSpeechSeq2Seq.from_pretrained("SeyiClover/whisper-small-ko", use_auth_token=HUGGINGFACE_TOKEN)
processor = AutoProcessor.from_pretrained("SeyiClover/whisper-small-ko-processor", use_auth_token=HUGGINGFACE_TOKEN)
tokenizer = AutoTokenizer.from_pretrained("SeyiClover/whisper-small-ko", use_auth_token=HUGGINGFACE_TOKEN)
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=HUGGINGFACE_TOKEN)

# 모델을 GPU로 이동 (가능한 경우)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
pipeline.to(device)

SR = 16000
CHUNK_DURATION = 30  # seconds

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 사용하지 않는 load_audio 함수 삭제

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

        # 같은 화자인 경우 정보 누적
        if speaker == current_speaker:
            current_segment['end'] = segment['end']
            current_segment['audio'] += segment['audio']

            # 전사 텍스트 있는 경우 텍스트도 누적
            if segment['transcription']:
                current_segment['transcription'] += f" {segment['transcription']}"

        else:
            # 한 화자의 발화 끝난 경우 누적된 정보를 merged_segment에 저장
            if current_segment:
                merged_segments.append(current_segment)
            current_speaker = speaker

            # 다른 화자의 첫 발화 current_segment 에 저장
            current_segment = {
                'speaker': speaker,
                'start': segment['start'],
                'end': segment['end'],
                'audio': segment['audio'],
                'transcription': segment['transcription'] # transcription 생성 후에도 화자병합 필요
            }

    if current_segment:
        merged_segments.append(current_segment)

    return merged_segments

def rename_speakers(segments):
    """화자 이름을 고유하게 재명명"""

    # 기존 함수에서 화자 이름이 순서대로 설정되지 않는 부분 수정
    # 화자를 숫자 부분 기준으로 추출 및 정렬
    unique_speakers = sorted(
        list({segment['speaker'] for segment in segments}),
        key=lambda x: int(x.split('_')[1])
    )
    # 화자 이름을 '참여자 1', '참여자 2' 등으로 매핑
    speaker_map = {speaker: f'참여자 {i+1}' for i, speaker in enumerate(unique_speakers)}

    # 모든 세그먼트에 대해 화자 이름 업데이트
    for segment in segments:
        segment['speaker'] = speaker_map[segment['speaker']]

    return segments

def speaker_diarize(path, num_speakers):
    try:

        # 오디오 파일 불러오고 전처리 
        base_path = os.path.splitext(path)[0] # 확장자 제외 경로
        temp_path = base_path + "_temp.wav" # 임시 경로

        cmd = [
            "ffmpeg", "-y", 
            "-threads", "0", # 모든 가능한 코어 사용
            "-i", path,
            "-f", "wav",
            "-ac", "1",
            "-acodec", "pcm_f32le",
            "-ar", str(SR),
            temp_path # temp_path 에 전처리 완료된 임시 파일 저장
        ]

        subprocess.run(cmd, check=True)

        # 화자분리 모델은 오디오 경로를 받아야 함
        # 전처리 완료된 오디오 경로 제공
        diarization = pipeline(temp_path, num_speakers=num_speakers)
        os.remove(temp_path) # 전처리 완료된 임시 파일은 화자분리 후 삭제 -> 원본 음성 파일/파일명 변화 없음
        
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
                'audio': audio_data.read(),
                'transcription': '' # transcription 생성 전후 모두 merge_speaker_segments 함수 사용하기 위해 추가
            })

        # STT 모델이 짧은 음성을 전사하는 경우 오류 자주 발생하므로, 같은 화자의 연속된 음성을 병합한 후 STT 전사 실행
        segments = merge_speaker_segments(segments)

        for segment in segments:
            audio_data = io.BytesIO(segment['audio'])
            segment['transcription'] = return_transcription(audio_data)

        # 'transcription' 결측치 제거
        # 'transcription' 값이 None, 빈 문자열, 혹은 공백만 있는 문자열인 경우 해당 요소를 제외하고 새로운 리스트 생성
        segments = [item for item in segments if item.get('transcription', '').strip() != '']

        # transcription 결측치 제거 후 화자 병합 재진행
        # 가독성 향상 위해 merge_speaker_segments 재실행하여 한 화자의 연속된 발화 합침
        segments = merge_speaker_segments(segments)
        segments = rename_speakers(segments)

        return segments  # 올바른 형식으로 반환
    
    except Exception as e:
        logging.error(f"오류 발생: {e}")
        raise
