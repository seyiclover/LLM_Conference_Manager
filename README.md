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
- **AI 모델**: OpenAI Whisper, Pytorch, Pyannote, HyperCLOVA X

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

## 프론트엔드 및 백엔드 구현


# 시연

### 데모 영상

- [영상 링크](https://youtu.be/0ExGMsiLkkE)




## Reference


