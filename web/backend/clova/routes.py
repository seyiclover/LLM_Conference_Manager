from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from common.models import get_db, Summary,User as UserModel
from .utils import STT_CompletionExecutor
import logging
from urllib.parse import urlparse
import os, json
from dotenv import load_dotenv
from ..auth.routes import get_current_user
from .utils import ChatCompletionExecutor, SummarizationExecutor, EmbeddingExecutor
from typing import Any, Dict
import asyncio

# milvus db
from pymilvus import Collection
from common.milvus import connect_to_milvus, get_milvus_collection

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
request_seg = os.getenv("REQUEST_SEG")

# 오픈소스 임베딩 api
request_emb_v2 = os.getenv("REQUEST_EMB_V2")
api_key_v2 = os.getenv("API_KEY_V2")
api_key_primary_val_v2 = os.getenv("API_KEY_PRIMARY_VAL_V2")


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

# 세션 상태 관리를 위한 딕셔너리
user_session_states: Dict[int, Dict[str, Any]] = {}

def get_session_state(user_id: int):

    prompt = "너는 사용자가 <회의에 대해 질문할 때>와 사용자가 <회의와 관련 없는 질문할 때> 를 구분하고, 다르게 행동하는 유능한 비서야. \n\
                - 답변은 친절하게, 한 줄로 짧게 해. 내용을 모두 전달한 후에는 {추가로 궁금한 점이 있으시면 말씀해 주세요.}라고 말해. \n\
                1. 사용자가 회의에 대해 질문하면, 네가 가지고있는 지식은 모두 배제하고, ## 회의 데이터 ## 만을 바탕으로 답변해. ## 회의 데이터 ## 는 사용자가 진행했던 회의 데이터야. \n\
                2. 사용자가 회의에 대해 질문한 게 아니라면, 반드시 ## 회의 데이터 ## 를 무시하고, 네가 가지고 있는 지식으로 친절하게 답해. \n\
                3. 사용자가 기존에 질문한 회의에 대한 추가 질문을 하면, 해당하는 과거 질문의 ## 회의 데이터 ## 를 바탕으로 답변해. \n\
            ## 예시 ## 의 비서처럼 답해. \n\
            ## 예시 ## \n\
                - 사용자:  서울시 여성 아동 외국인 회의 날짜가 언제였지? \n\
                - 비서: 2018년 11월 6일에 서울시 여성 아동 외국인 관련 시설 소관사항에 대한 2018년도 행정사무감사 실시를 선언한 회의가 진행되었습니다. \n\
                - 사용자: 회의에서 논의된 예산 관련 사항은 무엇인가요? \n\
                - 비서: 해당 회의에서 논의된 예산 관련 사항은 다음과 같습니다.\n\n- 영유아를 양육하는 아버지에게 강의형 또는 체험형의 교육을 진행하는 사업의 예산은 9997만 4000원이며, 이 중 8506만 원을 집행함 \n\
                - 사용자: 회의에서 나온 주요 이슈는 무엇인가요? \n\
                - 비서: 회의에서 나온 주요 이슈는 다음과 같습니다.\n\n 1. 서울여성공익센터와 서울여성공익센터 아리움의 주요 업무보고\n2. 서울시 육아종합지원센터의 역할과 노력\n3. 영유아를 양육하는 아버지에게 강의형 또는 체험형 교육을 진행하는 사업의 성과와 예산 집행 내역 \n\ "

    # 토큰 수 계산이 다르게 되는 것 같아서 삭제
    # chat_log, last_response, previous_messages 사용하지 않는데 헷갈려서 삭제
    if user_id not in user_session_states:
        user_session_states[user_id] = {
            "preset_messages": [{"role": "system", "content": prompt}],
            "summary_messages": [],
            "last_user_input": "",
            "last_user_message": {},
            "last_assistant_message": {},
            "system_message": prompt
        }
    return user_session_states[user_id]

def log_preset_messages(user_id: int):
    session_state = get_session_state(user_id)
    print(f"User {user_id} preset_messages:", json.dumps(session_state['preset_messages'], ensure_ascii=False, indent=2))
    print(f"User {user_id} system_message:", session_state['system_message'])

def summarize_and_reset(user_id: int, summarization_executor):
    session_state = get_session_state(user_id)

    # session_state['preset_messages'][0]은 프롬프트임
    # session_state['preset_messages']에서 프롬프트 제외하고 나머지 요약

    # msg['role'] == 'system': 챗봇에게 제공됐던 회의 데이터
    # msg['role'] == 'user', 'assistant': 사용자와 챗봇이 실제로 나눈 대화
    # --> 실제로 나눈 대화만 요약 (챗봇에게 제공됐던 회의 데이터는 요약하지 않음)
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

    # assistant message 가 생성되기 전 & 관련 회의 데이터 생성되기 전, user message 만 보낸 상태에서 max token 오류 발생함
    # 그러므로 벡터 검색 -> 관련 회의 데이터 생성을 위해 마지막 사용자 질문 반환 
    last_user_message = session_state.get('last_user_message', {}).get('content', '')

    return last_user_message

# Connect to Milvus
connect_to_milvus()

# Initialize EmbeddingExecutor
embedding_executor = EmbeddingExecutor(
        host=host,
        api_key_v2=api_key_v2,
        api_key_primary_val_v2=api_key_primary_val_v2
    )

# 사용자의 질문 임베딩하는 함수
def query_embed(text: str):
    request_data = {"text": text}
    response_data = embedding_executor.execute(request_data, request_id=request_emb_v2)
    return response_data

# 벡터 검색 후 결과 저장
# reference로 제공할 회의 데이터
# 유사도 거리는 챗봇에게 제공 안함
def vector_search(session_state, query_vector, collection):
    search_params = {"metric_type": "IP", "params": {"ef": 64}}
    results = collection.search(
        data=[query_vector],  # 검색할 벡터 데이터
        anns_field="embedding",  # 검색을 수행할 벡터 필드 지정
        param=search_params,
        limit=3,  # 관련도가 가장 높은 3개의 데이터 반환
        output_fields=["title", "date", "num_speakers", "text"]
    )

    for hit in results[0]:
        title = hit.entity.get("title")
        date = hit.entity.get("date")
        num_speakers = hit.entity.get("num_speakers")
        text = hit.entity.get("text")
        session_state['preset_messages'].append({
            "role": "system",
            "content": f"## 회의 데이터 ##\n- 회의 제목: {title}\n- 회의 날짜: {date}\n- 회의 참석자 수: {num_speakers}\n- 회의 일부: {text}\n"
        })


# 비동기 처리된 챗봇 엔드포인트
@router.post("/chat")
async def chat(question: Question, current_user: UserModel = Depends(get_current_user), collection: Collection = Depends(get_milvus_collection)):

    user_id = current_user.id
    session_state = get_session_state(user_id)

    # 사용자 쿼리 벡터화
    user_question = question.question
    query_vector = query_embed(user_question)

    # 벡터 검색 후 preset_messages에 결과 저장
    vector_search(session_state, query_vector, collection)
    
    session_state['last_user_input'] = user_question
    session_state['last_user_message'] = {"role": "user", "content": user_question}
    session_state['preset_messages'].append(session_state['last_user_message'])

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
    
    try:
        response = await asyncio.to_thread(completion_executor.execute, request_data, request_id=request_chat)
        response_text = response['result']['message']['content']
    
        session_state['last_assistant_message'] = {"role": "assistant", "content": response_text}
        session_state['preset_messages'].append(session_state['last_assistant_message'])

        log_preset_messages(user_id)


    except Exception as e:
        response_text = "죄송합니다. 채팅 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

        # 대화 생성 중 max token 제한 초과하는 경우 오류 발생하는 경우, 지금까지의 대화 요약
        try: 
            print(f"Error occured during chat for user {user_id}. Starting summarization.")

            # 반환된 마지막 사용자 질문 바탕으로 벡터 검색 수행
            last_user_question = summarize_and_reset(user_id, summarization_executor)
            query_vector = query_embed(last_user_question)
            vector_search(session_state, query_vector, collection)

            # preset_messages에 사용자 질문이 마지막에 와야 하므로 추가
            session_state['preset_messages'].append(session_state['last_user_message'])

            # request_data를 다시 생성 또는 업데이트
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

            log_preset_messages(user_id)

            response = await asyncio.to_thread(completion_executor.execute, request_data, request_id=request_chat)
            response_text = response['result']['message']['content']

            # 마지막 사용자 질문이 들어간 request_data 챗봇에게 전달 
            # -> max token 오류 발생
            # -> summarize_and_reset에서 요약 + 마지막 사용자 질문 preset_message 에 추가
            # -> response_text 다시 생성
            # -> 다시 생성된 response_text에서 챗봇 답변 추출해서 last_assistant_message, preset_message에 추가
            session_state['last_assistant_message'] = {"role": "assistant", "content": response_text}
            session_state['preset_messages'].append(session_state['last_assistant_message'])
            
        # max token 외 다른 오류 발생시
        except Exception as e:
            logger.exception(f"Error occurred during chat for user {user_id}: {e}")
            response_text = "죄송합니다. 채팅 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

    return {"response": response_text}