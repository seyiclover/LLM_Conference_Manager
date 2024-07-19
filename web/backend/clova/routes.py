from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from common.models import get_db, Summary, Transcript, File  ,User as UserModel
from .utils import STT_CompletionExecutor
import logging
from urllib.parse import urlparse
import os, json
from dotenv import load_dotenv
from ..auth.routes import get_current_user
from .utils import ChatCompletionExecutor, SummarizationExecutor
from konlpy.tag import Okt
from typing import Any, Dict
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import asyncio ,time
import yappi




# .env 파일에서 환경 변수 로드
load_dotenv(dotenv_path='../.env')

router = APIRouter()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 사용
host_url = os.getenv("HOST")
api_key = os.getenv("API_KEY")
api_key_primary_val = os.getenv("API_KEY_PRIMARY_VAL")
request_sum = os.getenv("REQUEST_SUM")
request_chat = os.getenv("REQUEST_CHAT")

# 호스트 추출
parsed_url = urlparse(host_url)
host = parsed_url.hostname

class Question(BaseModel):
    question: str

# CompletionExecutor 인스턴스 생성
stt_completion_executor = STT_CompletionExecutor(host, api_key, api_key_primary_val, request_sum)

# 요청 데이터 모델
class RequestDataModel(BaseModel):
    text: str
    transcript_id: int
    segMinSize: int = 100
    includeAiFilters: bool = True
    autoSentenceSplitter: bool = True
    segCount: int = 4
    segMaxSize: int = 3000

# STT 요약 API
@router.post("/summarize")
async def summarize_text(request_data: RequestDataModel = Body(...), db: Session = Depends(get_db)):
    # DB에서 요약 데이터 확인
    summary = Summary.get_summary_by_transcript_id(db, request_data.transcript_id)
    
    # 요약 데이터가 존재하면 DB에서 반환
    if summary:
        return {"summary": summary.summary_text}

    # 존재하지 않으면 API 호출
    completion_request = {
        "texts": [request_data.text],
        "segMinSize": request_data.segMinSize,
        "includeAiFilters": request_data.includeAiFilters,
        "autoSentenceSplitter": request_data.autoSentenceSplitter,
        "segCount": request_data.segCount,
        "segMaxSize": request_data.segMaxSize
    }
    response_text = stt_completion_executor.execute(completion_request)
    
    # 요약 데이터 DB에 저장
    new_summary = Summary(transcript_id=request_data.transcript_id, summary_text=response_text)
    db.add(new_summary)
    db.commit()

    return {"summary": response_text}


# 형태소 분석기 초기화
okt = Okt()



def preprocess_text(text):
    pos_tags = okt.pos(text)
    result = []
    for word, tag in pos_tags:
        if tag not in ['Josa', 'Conjunction', 'Eomi', 'Exclamation']:
            result.append(word)
    return ' '.join(result)

user_session_states: Dict[int, Dict[str, Any]] = {}

def get_session_state(user_id: int):
    if user_id not in user_session_states:
        user_session_states[user_id] = {
            "preset_messages": [],
            "total_tokens": 0,
            "chat_log": [],
            "started": False,
            "summary_messages": [],
            "last_user_input": "",
            "last_response": "",
            "last_user_message": {},
            "last_assistant_message": {},
            "previous_messages": [],
            "system_message_changed": False,
            "system_message": "system_message = 너는 사용자가 <회의에 대해 질문할 때>와 사용자가 <회의와 관련 없는 질문할 때> 를 구분하고, 다르게 행동하는 유능한 비서야. \n\
                        - 답변은 한 줄로 짧게 해. 내용을 모두 전달한 후에는 {추가로 궁금한 점이 있으시면 말씀해 주세요.}라고 말해. \n\
                        1. 사용자가 회의에 대해 질문하면, 네가 가지고있는 지식은 모두 배제하고, ## 회의 데이터 ## 만을 바탕으로 답변해. ## 회의 데이터 ## 는 사용자가 진행했던 회의 데이터야. \n\
                        2. 사용자가 회의에 대해 질문한 게 아니라면, 반드시 ## 회의 데이터 ## 를 무시하고, 네가 가지고 있는 지식으로 친절하게 답해. \n\
                        3. 사용자가 기존에 질문한 회의에 대한 추가 질문을 하면, 해당하는 과거 질문의 ## 회의 데이터 ## 를 바탕으로 답변해. \n\
                            유능한 비서의 답변 ## 예시 ## 를 참고해. \n\
                            - 질문1:  서울시 여성 아동 외국인 회의 날짜가 언제였지? \n\
                            - 답변1: 2018년 11월 6일에 서울시 여성 아동 외국인 관련 시설 소관사항에 대한 2018년도 행정사무감사 실시를 선언한 회의가 진행되었습니다. \n\
                            - 질문2: 회의에서 논의된 예산 관련 사항은 무엇인가요? \n\
                            - 답변2: 해당 회의에서 논의된 예산 관련 사항은 다음과 같습니다.\n\n- 영유아를 양육하는 아버지에게 강의형 또는 체험형의 교육을 진행하는 사업의 예산은 9997만 4000원이며, 이 중 8506만 원을 집행함 \n\
                            - 질문3: 회의에서 나온 주요 이슈는 무엇인가요? \n\
                            - 답변3: 회의에서 나온 주요 이슈는 다음과 같습니다.\n\n 1. 서울여성공익센터와 서울여성공익센터 아리움의 주요 업무보고\n2. 서울시 육아종합지원센터의 역할과 노력\n3. 영유아를 양육하는 아버지에게 강의형 또는 체험형 교육을 진행하는 사업의 성과와 예산 집행 내역 \n\ "
        }
    return user_session_states[user_id]

def log_preset_messages(user_id: int):
    session_state = get_session_state(user_id)
    print(f"User {user_id} preset_messages:", json.dumps(session_state['preset_messages'], ensure_ascii=False, indent=2))
    print(f"User {user_id} system_message:", session_state['system_message'])
    print(f"User {user_id} total_tokens:", session_state['total_tokens'])

def summarize_and_reset(user_id: int, summarization_executor):
    session_state = get_session_state(user_id)
    text_to_summarize = " ".join([msg.get('content', '') for msg in session_state['preset_messages'][1:] if msg['role'] != 'system'])
    print(f"Text to summarize for user {user_id}: {text_to_summarize}")

    summary_request_data = {
        "texts": [text_to_summarize],
        "autoSentenceSplitter": True,
        "segCount": -1,
        "segMaxSize": 1000,
        "segMinSize": 300,
        "includeAiFilters": False
    }

    summary_response = summarization_executor.execute(summary_request_data, request_id=request_sum)
    print(f"Summary response for user {user_id}: {summary_response}")

    if summary_response and isinstance(summary_response, str):
        summary_text = summary_response
        session_state['summary_messages'].append({"role": "system", "content": summary_text})

        session_state['preset_messages'] = [{"role": "system", "content": session_state['system_message']}, {"role": "system", "content": summary_text}]
        
        session_state['preset_messages'].append(session_state.get('last_user_message', {}))
        session_state['preset_messages'].append(session_state.get('last_assistant_message', {}))

        session_state['total_tokens'] = len(summary_text.split()) + \
            len(session_state.get('last_user_message', {}).get('content', '').split()) + \
            len(session_state.get('last_assistant_message', {}).get('content', '').split())

    log_preset_messages(user_id)
    
@lru_cache(maxsize=128)
def get_transcripts_from_db(db: Session, user_id: int):
    transcripts = db.query(Transcript).join(File).filter(File.user_id == user_id).all()
    data = {
        "title": [],
        "date": [],
        "content": [],
        "transcript_id": []
    }

    for transcript in transcripts:
        data["title"].append(transcript.title)
        data["date"].append(transcript.date)
        data["content"].append(transcript.content)
        data["transcript_id"].append(transcript.id)

    return pd.DataFrame(data)

def get_summary_from_db(db: Session, transcript_id: int):
    summary = db.query(Summary).filter(Summary.transcript_id == transcript_id).first()
    return summary.summary_text if summary else "요약 없음"

def get_most_similar_meeting(user_question, df, vectorizer, tfidf_matrix):
    question_tfidf = vectorizer.transform([user_question])
    cosine_similarities = cosine_similarity(question_tfidf, tfidf_matrix)
    most_similar_index = cosine_similarities.argmax()
    return df.iloc[most_similar_index]


# 비동기 처리된 챗봇 엔드포인트
@router.post("/chat")
async def chat(question: Question, db: Session = Depends(get_db), current_user: UserModel = Depends(get_current_user)):

    user_id = current_user.id
    session_state = get_session_state(user_id)

    user_question = question.question
    df = get_transcripts_from_db( db, user_id)

    df['date'] = df['date'].astype(str)
    
    vectorizer = TfidfVectorizer(max_df=0.95, ngram_range=(1, 2), sublinear_tf=True, norm='l2', preprocessor=preprocess_text)
    df['combined'] = "회의 제목: " + df['title'] + ", 회의 날짜: " + df['date'] + ", 회의 내용: " + df['content']
    
    tfidf_matrix = vectorizer.fit_transform(df['combined'])
    
    relevant_meeting = get_most_similar_meeting(user_question, df, vectorizer, tfidf_matrix)

    summary_text = await asyncio.to_thread(get_summary_from_db, db, relevant_meeting['transcript_id'])
    
    session_state['last_user_input'] = user_question
    session_state['last_user_message'] = {"role": "user", "content": user_question}
    
    session_state['preset_messages'].append({
        "role": "system", 
        "content": f"## 회의 데이터 ## 회의 제목: {relevant_meeting['title']}, 회의 날짜: {relevant_meeting['date']}, 회의 요약본:{summary_text}"
    })
    session_state['preset_messages'].append(session_state['last_user_message'])
    session_state['chat_log'].append(session_state['last_user_message'])

    request_data = {
        "messages": session_state['preset_messages'],
        "maxTokens": 256,
        "temperature": 0.5,
        "topK": 0,
        "topP": 0.6,
        "repeatPenalty": 1.2,
        "stopBefore": [],
        "includeAiFilters": True,
        "seed": 0,
    }

    try:
        completion_executor = ChatCompletionExecutor(
            host=host,
            api_key=api_key,
            api_key_primary_val=api_key_primary_val
        )
        summarization_executor = SummarizationExecutor(
            host=host,
            api_key=api_key,
            api_key_primary_val=api_key_primary_val
        )

        response = await asyncio.to_thread(completion_executor.execute, request_data, request_id=request_chat)
       
        response_text = response['result']['message']['content']
        
        session_state['last_response'] = response_text
        session_state['last_assistant_message'] = {"role": "assistant", "content": response_text}

        session_state['preset_messages'].append(session_state['last_assistant_message'])
        session_state['chat_log'].append(session_state['last_assistant_message'])

        session_state['total_tokens'] = len(" ".join(msg['content'] for msg in session_state['preset_messages']).split())

        log_preset_messages(user_id)

        token_limit = 4096 - request_data["maxTokens"]
        if session_state['total_tokens'] > token_limit:
            print(f"Token limit exceeded for user {user_id}. Starting summarization.")
            await asyncio.to_thread(summarize_and_reset, user_id, summarization_executor)

    except Exception as e:
        logger.exception(f"Error occurred during chat for user {user_id}: {e}")
        response_text = "죄송합니다. 채팅 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

    return {"response": response_text}