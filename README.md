# LLM Conference Manager
고용노동부-대한상공회의소 주관 미래내일 일경험 프로젝트

![Nexo_logo2_name](https://github.com/user-attachments/assets/a87d7a88-9629-4ae7-895b-3b2f07322fa3)

# NexoChat

LLM을 활용한 사용자 콘텐츠 기반의 컨퍼런스 매니저

## 소개
NexoChat은 초거대 언어모델(LLM)을 활용하여 사용자 맞춤형 기록 시스템을 제공합니다. 장기간 누적된 회의 데이터를 챗봇 형식으로 검색 및 질의응답할 수 있으며, 컨퍼런스 매니징 기능을 포함하고 있습니다.

### 주요 기능
- **음성 처리 챗봇**: Hyper CLOVA X API 사용
- **회의록 요약 및 검색**
- **회의록 기반 질의응답**
- **화자 분리 및 음성 인식 (STT)**

### 개발 인원 및 기간

- 개발 기간 : 2024/5/13 ~ 2024/7/12
- 개발 인원
    - ai : 이세이, 박지호
    - 프론트,백엔드 : 박건우

## 시스템 구조

<img width="712" alt="image" src="https://github.com/user-attachments/assets/9ec1022d-dd71-45f5-b05f-a4d483f0ba28">


### 기술 스택
- **프론트엔드**: React
- **백엔드**: FastAPI, MySQL, SQLAlchemy, Alembic
- **디자인**: Figma
- **AI**: Python, Huggingface, Pytorch
- **사용모델**: OpenAI Whisper, Pyannote, HyperCLOVA X api

## 프론트엔드 및 백엔드 구현

### 프론트엔드
1. **Google OAuth 로그인**
   - 사용자 편의성 증대 및 보안 강화
   - Google OAuth API를 통한 인증 후 백엔드에서 JWT 발급
2. **JWT 토큰 관리**
   - 로그인 성공 시 JWT를 로컬 스토리지에 저장
   - API 요청 시 토큰을 헤더에 포함하여 인증
3. **파일 업로드**
   - FormData 객체를 사용하여 파일 및 메타데이터 전송
   - 백엔드에서 파일 처리 및 텍스트 변환

### 백엔드
1. **인증 시스템**: Google OAuth를 통한 로그인, JWT 토큰 생성 및 관리
2. **사용자 관리**: 사용자 정보 관리 및 조회
3. **음성 파일 업로드 및 처리**: 화자 분리, 음성 -> 텍스트 변환
4. **데이터 처리**: 파일 업로드 처리, AI 모델을 통한 텍스트 유사도 분석 및 응답 생성

## AI 기능 구현 
<img width="749" alt="image" src="https://github.com/user-attachments/assets/4441aa48-f95a-42ac-9194-4dad65f93758">

### 1. STT 
- 한국어 대용량 데이터셋으로 학습된 pretrained whisper small 모델을 자체 구축 데이터셋으로 파인튜닝하여 사용
- base model: SungBeom/whisper-small-ko
  
  https://huggingface.co/SungBeom/whisper-small-ko#training-hyperparameters
  
- 파인튜닝 사용 데이터셋
  
  <img width="626" alt="image" src="https://github.com/user-attachments/assets/083bec6d-7008-4d7a-9160-adb5abcb2805">
  
- 파인튜닝 학습 성능
  
    | Training Loss | Epoch  | Step  | Cer     | Validation Loss | Wer     |
    |:-------------:|:------:|:-----:|:-------:|:---------------:|:-------:|
    | 0.3849        | 5.5617 | 10000 |  9.9827 | 0.3555          | 25.0944 |
  <img width="786" alt="image" src="https://github.com/user-attachments/assets/4b6d38ab-deab-4401-b6d5-9854bd2ab1f1">

#### Python Usage
```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer

model = WhisperForConditionalGeneration.from_pretrained("NexoChatFuture/whisper-small-youtube-extra")
feature_extractor = WhisperFeatureExtractor.from_pretrained("NexoChatFuture/whisper-small-youtube-extra")
tokenizer = WhisperTokenizer.from_pretrained("NexoChatFuture/whisper-small-youtube-extra-tokenizer")
processor = WhisperProcessor.from_pretrained("NexoChatFuture/whisper-small-youtube-extra-processor")

```
 
### 2. 화자 분리  
- pyannote speaker diarization 오픈 소스 모델 사용
  
  https://github.com/pyannote/pyannote-audio

### 3. 챗봇
- Naver HyperCLOVA X api 사용 챗봇 기능 구현
- 핵심 기능
  1) 회의록 요약
  2) 회의 데이터 기반 검색 및 질의응답 

## 웹 페이지 화면 
<table>
  <tr>
    <td><img width="752" height="270" alt="image" src="https://github.com/user-attachments/assets/5268fa56-1c56-4792-92f1-af7c5804ec3f"></td>
    <td><img width="752" height="270" alt="image" src="https://github.com/user-attachments/assets/21d89188-4d97-4466-a6db-d71b4242c1f1"></td>
  </tr>
</table>


## 시연

### 데모 영상

- [영상 링크](https://youtu.be/0ExGMsiLkkE)




## Reference


