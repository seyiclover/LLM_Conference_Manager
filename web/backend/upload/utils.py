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
API_URL = "https://api-inference.huggingface.co/models/svenskpotatis/whisper-small-youtube-extra"
headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}

SR = 16000
CHUNK_DURATION = 30  # seconds

# 로깅 설정
logging.basicConfig(level=logging.INFO)

def to_wav(path):
    ''' 오디오/비디오 파일을 wav로 변환 '''
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
    """ 임시 WAV 파일 삭제"""
    os.remove(wav_path)  # 파일 삭제
    print(f"Temporary file {wav_path} has been deleted.")

def return_transcription(audio_start, audio_end, file_path):
    """오디오 데이터를 전사하여 텍스트 반환"""
    try:
        transcriptions = []

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
            chunk, _ = process.communicate()

            response = requests.post(API_URL, headers=headers, data=chunk)

            try:
                transcription = json.loads(response.content)['text']
                # print(transcription)
                transcriptions.append(transcription)
            except json.JSONDecodeError:
                print(f"Error decoding JSON for chunk {chunk_start//CHUNK_DURATION + 1}: {response.content}")
            except KeyError:
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

# segmentation 모델
class CustomPyannoteONNX(PyannoteONNX):
    def __init__(self, show_progress=False, inter_threads=1, intra_threads=1):
        super().__init__(show_progress=show_progress)
        self.session = self._create_custom_session(inter_threads, intra_threads)

    def _create_custom_session(self, inter_threads, intra_threads):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = inter_threads
        opts.intra_op_num_threads = intra_threads
        model_path = '../models/segmentation-3.0.onnx' # segmentation 모델 onnx
        return ort.InferenceSession(model_path, sess_options=opts)

# 화자분리 pipeline: segmentation -> embedding -> clustering
class DiarizePipeline:
    def __init__(self, audio_file, num_speakers=3):
        self.audio_file = audio_file
        self.num_speakers = num_speakers
        self.sample_rate = 16000  
        self.cores = multiprocessing.cpu_count() - 1 # 최대 코어 수-1개 사용
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
            if track['stop'] - track['start'] > 1: # 1초 이상 오디오만 처리 (whisper hallucination, embedding 오류 방지)
                seg_info.append(track)
        self.segments = seg_info
        return seg_info

    # embedding
    def compute_embeddings(self):
        embeddings = []
        for segment in tqdm(self.segments):
            start_time = segment["start"]
            end_time = segment["stop"]
            emb = self.embed_main(self.audio_file, start_time, end_time)
            embeddings.append(emb)
        self.embeddings = np.array(embeddings).squeeze()
        return self.embeddings

    def embed_main(self, wav_path, start_time, end_time):
        onnx_path = '../models/voxceleb_resnet34_LM.onnx'
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.cores
        so.inter_op_num_threads = self.cores
        session = ort.InferenceSession(onnx_path, sess_options=so)
        feats = self.compute_fbank(wav_path, start_time=start_time, end_time=end_time)
        feats = feats.unsqueeze(0).numpy()  # add batch dimension
        embeddings = session.run(output_names=['embs'], input_feed={'feats': feats})
        return embeddings[0]

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
        self.diarization()
        self.compute_embeddings()
        self.cluster_speakers()
        return merge_speaker_segments(self.segments)
