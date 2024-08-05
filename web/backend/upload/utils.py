import numpy as np
import io, os
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer
import subprocess
import logging

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from pyannote_onnx import PyannoteONNX
import onnxruntime as ort
import librosa
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import multiprocessing
import subprocess
from concurrent.futures import ThreadPoolExecutor

import requests, json

# .env 파일 로드
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Hugging Face token
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# 모델의 API URL (public 이어야 함)
# 최종 모델로 변경 필요
API_URL = "https://api-inference.huggingface.co/models/svenskpotatis/whisper-small-youtube-extra"
headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}

SR = 16000
CHUNK_DURATION = 30  # seconds

# 로깅 설정
logging.basicConfig(level=logging.INFO)

def to_wav(path):
    ''' 오디오/비디오 파일을 wav로 변환 

    Parameters
    ---
    path: 사용자가 업로드한 오디오/비디오 파일 경로 (ex. audio.mp3)

    Returns
    ---
    wav_path: wav로 변환된 파일 경로 (ex. audio_temp.wav)
    '''
    base_path = os.path.splitext(path)[0] # 확장자 제외 경로
    wav_path = base_path + "_temp.wav" # 임시 경로

    cmd = [
        "ffmpeg", "-y", 
        "-threads", "0", # 모든 가능한 코어 사용
        "-i", path,
        "-f", "wav",
        "-ac", "1",
        "-acodec", "pcm_f32le",
        "-ar", '16000',
        wav_path # 전처리 파일 경로
    ]

    subprocess.run(cmd, check=True)

    return wav_path

def delete_file(wav_path):
    """ 임시 WAV 파일 삭제 """
    os.remove(wav_path)  # 파일 삭제
    print(f"Temporary file {wav_path} has been deleted.")

def return_transcription(audio_start, audio_end, file_path):
    """ 오디오 데이터를 전사하여 텍스트 반환 
    
    Parameters
    ---
    audio_start: 발화 시작시간
    audio_end: 발화 끝 시간
    file_path: 오디오 경로

    Returns
    ---
    transcriptions: 전사된 텍스트 string
    """
    try:
        transcriptions = []
        
        # whisper 가 전사할 수 있는 음성은 최대 30초이므로
        # CHUNK_DURATION(30초) 단위로 나누어 순차적으로 전사
        for chunk_start in range(int(audio_start), int(audio_end)+1, CHUNK_DURATION):

            chunk_end = min(chunk_start + CHUNK_DURATION, int(audio_end)+1)

            cmd = [
                    "ffmpeg", "-nostdin",
                    "-i", file_path,
                    "-ss", str(chunk_start),
                    "-to", str(chunk_end),
                    "-f", "wav",
                    "-ac", "1",
                    "-ar", str(SR),
                    "pipe:1"
                ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            chunk, _ = process.communicate() # chunk: 30초 단위로 분할된 음성

            # huggingface inference api 사용
            # data: wav 파일 형식
            response = requests.post(API_URL, headers=headers, data=chunk)

            try:
                transcription = json.loads(response.content)['text']
                # print(transcription)
                transcriptions.append(transcription)
            except json.JSONDecodeError:
                print(f"Error decoding JSON for chunk {chunk_start//CHUNK_DURATION + 1}: {response.content}")
            except KeyError:
                # 첫 api 호출시 에러 - 해결 필요
                # {"error":"Model svenskpotatis/whisper-small-youtube-extra is currently loading","estimated_time":38.67758560180664}
                print(f"No 'text' key in response for chunk {chunk_start//CHUNK_DURATION + 1}: {response.content}")

        print(transcriptions)
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
            current_segment['stop'] = segment['stop']

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
                'stop': segment['stop'],
                'transcription': segment['transcription'] 
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

# 화자분리 pipeline: 
# 1. segmentation: 음성에서 발화부분 분리 
# 2. embedding: 분리된 각 발화 구간에서 embedding 추출
# 3. clustering: 추출된 embedding을 지정된 화자 수(num_speakers)에 맞게 clustering

class CustomPyannoteONNX(PyannoteONNX):
    '''Segmetation 모델: Pyannote segmentation 모델 상속받아 수정

    기존 PyannoteONNX 모델 기능 확장 -> 사용할 CPU 코어 수 조정할 수 있게 하여 성능, 리소스 최적화

    Attributes
    ----------
    session (ort.InferenceSession) : ONNX 모델 추론 세션. 사용자 정의 스레드 설정 적용
    
    Methods
    -------
    _create_custom_session(self, inter_threads, intra_threads) : 사용자 정의 추론 세션 생성
        inter_threads : 연산 간 병렬처리 - 여러 연산을 동시에 실행하는 데 사용되는 스레드 수 
        intra_threads : 연산 내 병렬처리 - 단일 연산을 수행할 때 사용되는 스레드 수
        model_path : segmentation model (segmentation-3.0.onnx) 경로
    '''
    def __init__(self, show_progress=False, inter_threads=1, intra_threads=1):
        super().__init__(show_progress=show_progress)
        self.session = self._create_custom_session(inter_threads, intra_threads)

    def _create_custom_session(self, inter_threads, intra_threads):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = inter_threads
        opts.intra_op_num_threads = intra_threads
        model_path = '../models/segmentation-3.0.onnx' # segmentation onnx 모델
        return ort.InferenceSession(model_path, sess_options=opts)


class DiarizePipeline:
    '''화자분리 Pipeline
    
    Segmentation(segmentation-3.0.onnx) > Embedding(voxceleb_resnet34_LM.onnx) > Clustering(AgglomerativeClustering)
    Pyannote.audio 의 speaker-diariazation pipeline과 모델, 알고리즘은 동일하나 세부적인 코드는 다르므로 성능 차이 있을 수 있음

    Parameters
    ----------
    audio_file : 오디오 파일 경로
    num_speakers : 화자 수

    Returns
    -------
    화자분리 및 병합 결과
        ex. [{'speaker': '참석자 1', 'start': 0.062, 'stop': 20.447, 'transcription': ''},
             {'speaker': '참석자 2', 'start': 20.565, 'stop': 24.294, 'transcription': ''}, ...]
    
    '''
    
    def __init__(self, audio_file, num_speakers=3):
        self.audio_file = audio_file # 오디오 파일 경로
        self.num_speakers = num_speakers # 화자 수
        self.sample_rate = 16000 
        self.cores = multiprocessing.cpu_count() - 1 # 최대 cpu 코어 수-1개 사용
        self.pyannote_onnx = self.init_pyannote()
        self.segments = []
        self.embeddings = []
        self.labels = []

    # segmentation
    def init_pyannote(self):
        return CustomPyannoteONNX(show_progress=True, inter_threads=self.cores, intra_threads=self.cores)

    def load_audio(self):
        self.wav, _ = librosa.load(self.audio_file, sr=self.sample_rate, mono=True)
        return self.wav

    def diarization(self):
        seg_info = []
        for track in self.pyannote_onnx.itertracks(self.wav):
            # 1초 이상 오디오만 처리 
            # 너무 짧은 음성에서 whisper hallucination, embedding 오류 발생
            if track['stop'] - track['start'] > 1: 
                seg_info.append(track)
        self.segments = seg_info
        return seg_info

    # embedding
    '''
    오디오에서 Mel-scale fbank 특성 추출 > 추출된 특성의 임베딩 반환
    '''
    def compute_embeddings(self):
        embeddings = []
        onnx_path = '../models/voxceleb_resnet34_LM.onnx'  # embedding 모델 경로
        
        # ONNX 세션 설정
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.cores
        so.inter_op_num_threads = self.cores
        session = ort.InferenceSession(onnx_path, sess_options=so)
        
        # 모든 발화 segment에 대해 반복
        for segment in tqdm(self.segments):
            start_time = segment["start"]
            end_time = segment["stop"]
            
            # 오디오에서 fbank 특성 추출
            feats = self.compute_fbank(self.audio_file, start_time=start_time, end_time=end_time)
            feats = feats.unsqueeze(0).numpy()  # 배치 차원 추가
            
            # 임베딩 계산
            emb = session.run(output_names=['embs'], input_feed={'feats': feats})[0]
            embeddings.append(emb)
        
        self.embeddings = np.array(embeddings).squeeze()
        return self.embeddings  # embedding 정보 반환

    # 오디오에서 fbank 특성 추출
    def compute_fbank(self, wav_path, start_time=None, end_time=None,
                      num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0):
        waveform, sample_rate = torchaudio.load(wav_path)
        if start_time is not None and end_time is not None:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            waveform = waveform[:, start_sample:end_sample]
        waveform = waveform * (1 << 15)
        mat = torchaudio.compliance.kaldi.fbank(
            waveform,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            sample_frequency=sample_rate,
            window_type='hamming',
            use_energy=False
        )
        mat = mat - torch.mean(mat, dim=0)
        return mat

    # clustering
    def cluster_speakers(self):
        clustering = AgglomerativeClustering(self.num_speakers).fit(self.embeddings)
        self.labels = clustering.labels_
        for i, segment in enumerate(self.segments):
            segment["speaker"] = '참석자 '+ str(self.labels[i] + 1)
            segment['transcription'] = ''
        return self.labels

    def run(self):
        self.load_audio()
        self.diarization() # segmentation
        self.compute_embeddings() # embedding
        self.cluster_speakers() # clustering
        return merge_speaker_segments(self.segments) # 같은 화자의 연속된 발화 병합
