import os
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException, Body
from fastapi.responses import JSONResponse
from common.models import get_db, File as FileModel, User as UserModel, Transcript as TranscriptModel
from backend.auth.routes import get_current_user
from backend.upload.utils import speaker_diarize

# milvus
from common.milvus import connect_to_milvus, check_and_create_collection
from langchain_core.documents.base import Document
from clova.utils import SegmentationExecutor, EmbeddingExecutor
import json
import time
from urllib.parse import urlparse

router = APIRouter()

# 환경 변수 사용
host_url = os.getenv("HOST")
api_key = os.getenv("API_KEY")
api_key_primary_val = os.getenv("API_KEY_PRIMARY_VAL")
request_sum = os.getenv("REQUEST_SUM")
request_chat = os.getenv("REQUEST_CHAT")
request_emb = os.getenv("REQUEST_EMB")
request_seg = os.getenv("REQUEST_SEG")

# 호스트 추출
parsed_url = urlparse(host_url)
host = parsed_url.hostname


class FileUploadRequest(BaseModel):
    file: UploadFile
    meeting_name: str = Field(..., description="회의 이름")
    meeting_date: str = Field(..., description="회의 날짜", pattern=r'^\d{4}-\d{2}-\d{2}$')
    speaker_count: int = Field(..., description="발화자 수", ge=1)

class TranscriptResponse(BaseModel):
    id: int
    file_id: int
    title: str
    date: str
    num_speakers: int
    content: str
    created_at: str

class AudioToTextRequest(BaseModel):
    file_id: str
    num_speakers: int
    title: str
    meeting_date: str

@router.post("/audioToText")
async def diarize_and_transcribe(
    payload: AudioToTextRequest = Body(...),
    db: Session = Depends(get_db)
):
    try:
        db_file = db.query(FileModel).filter(FileModel.id == payload.file_id).first()
        if not db_file:
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없음")

        file_path = db_file.file_path
        logging.info(f"파일 경로: {file_path}")

        segments = speaker_diarize(file_path, payload.num_speakers)

        # segments 형식 검증
        if isinstance(segments, list) and all(isinstance(segment, dict) for segment in segments):
            transcript_text = "\n".join([f"{segment['speaker']}: {segment['transcription']}" for segment in segments])
        else:
            logging.error(f"Invalid format from speaker_diarize")
            raise ValueError("Invalid format from speaker_diarize")

        meeting_date_dt = datetime.strptime(payload.meeting_date, '%Y-%m-%d')

        transcript = TranscriptModel(
            file_id=payload.file_id,
            title=payload.title,
            date=meeting_date_dt,
            num_speakers=payload.num_speakers,
            content=transcript_text  # 텍스트로 저장
        )
        db.add(transcript)
        db.commit()
        db.refresh(transcript)

        logging.info(f"Generated transcript text: {transcript_text}")

        # milvus db 에 데이터 저장
        process_and_embed_transcript(transcript)

        return JSONResponse(content={"message": "Transcript successfully created","meeting_id": transcript.id}, status_code=200)

    except Exception as e:
        logging.error(f"오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/files")
async def upload_file(
    file: UploadFile = File(...),
    meeting_name: str = Form(...),
    meeting_date: str = Form(...),
    speaker_count: int = Form(...),
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        user_id = current_user.id
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../common/uploaded_files'))
        user_folder = os.path.join(base_dir, str(user_id))
        os.makedirs(user_folder, exist_ok=True)

        original_filename = file.filename
        filename = f"{meeting_name}{os.path.splitext(original_filename)[1]}"
        file_path = os.path.join(user_folder, filename)

        with open(file_path, "wb") as f:
            await file.seek(0)
            f.write(await file.read())

        meeting_date_dt = datetime.strptime(meeting_date, '%Y-%m-%d')

        db_file = FileModel(
            filename=filename,
            user_id=user_id,
            speaker_count=speaker_count,
            meeting_date=meeting_date_dt,
            file_path=file_path
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)


        return {"message": "파일 업로드 성공", "filename": db_file.filename, "id": db_file.id}
    except Exception as e:
        db.rollback()
        logging.error(f"오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#파일목록 불러오는 api
@router.get("/listMeetings")
async def list_meetings(current_user: UserModel = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        user_id = current_user.id
        meetings = db.query(FileModel).filter(FileModel.user_id == user_id).all()
        # 날짜 포맷팅하여 시간 제거 및 파일 확장자 제거
        formatted_meetings = [
            {
                "id": meeting.id,
                "filename": meeting.filename.rsplit('.', 1)[0],  # 확장자 제거
                "meeting_date": meeting.meeting_date.strftime('%Y-%m-%d')  # 날짜만 포맷팅
            }
            for meeting in meetings
        ]
        return {"meetings": formatted_meetings}
    except Exception as e:
        logging.error(f"오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/meeting/{meeting_id}", response_model=TranscriptResponse)
async def get_meeting_detail(meeting_id: int, db: Session = Depends(get_db)):
    try:
        logging.info(f"Requested meeting_id: {meeting_id}")
        # 회의록 조회
        transcript = db.query(TranscriptModel).filter(TranscriptModel.id == meeting_id).first()
        if not transcript:
            logging.error(f"회의록을 찾을 수 없습니다. meeting_id: {meeting_id}")
            raise HTTPException(status_code=404, detail="회의록을 찾을 수 없습니다.")

        response = TranscriptResponse(
            id=transcript.id,
            file_id=transcript.file_id,
            title=transcript.title,
            date=transcript.date.strftime('%Y-%m-%d'),
            num_speakers=transcript.num_speakers,
            content=transcript.content,
            created_at=transcript.created_at.strftime('%Y-%m-%d %H:%M:%S')
        )

        
        return response
    except Exception as e:
        logging.error(f"오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# 업로드된 회의 데이터 milvus db에 저장
# 회의 텍스트는 문단 나눠서 분할 후 임베딩 -> db 저장
def process_and_embed_transcript(transcript: TranscriptModel):

    # 업로드된 회의 데이터
    content = transcript.content
    title = transcript.title
    meeting_date = transcript.date.strftime('%Y-%m-%d')
    num_speakers = transcript.num_speakers
    
    # 텍스트를 Document 객체로 변환
    # 텍스트 데이터(회의 텍스트)와 메타데이터(날짜 등)를 효율적으로 관리 위함
    meeting_data = [Document(page_content=content, metadata={'title': title, 'date': meeting_date, 'num_speakers': num_speakers})]
    print("Converted transcript to Document and segment")

    ######################################################
    # 문단 나누기
    ######################################################

    # 문단 나누기 api
    segmentation_executor = SegmentationExecutor(
            host=host,
            api_key=api_key,
            api_key_primary_val=api_key_primary_val
        )

    chunked_data = []

    for data in meeting_data:
        try:
            request_data = {
                "alpha": -100,
                "segCnt": -1,
                "postProcessMinSize": 500,
                "text": data.page_content,
                "postProcess": False
            }

            request_json_string = json.dumps(request_data)
            request_data = json.loads(request_json_string, strict=False)
            response_data = segmentation_executor.execute(request_data, request_id=request_seg)
            result_data = [' '.join(segment) for segment in response_data]
            print(result_data)

        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding failed: {e}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")

        for paragraph in result_data:
            chunked_document = {
                "title": data.metadata["title"],
                "date": data.metadata["date"],
                "num_speakers": data.metadata["num_speakers"],
                "text": paragraph
            }
            chunked_data.append(chunked_document)
    
    print("문단 나누기 완료")
    
    # milvus db 연결
    connect_to_milvus()
    collection = check_and_create_collection('meeting_data')

    ######################################################
    # 임베딩 생성 후 저장
    ######################################################

    # too many requests 가끔 걸렸음
    # 임베딩 재시도 간격과 최대 재시도 횟수 설정
    RETRY_DELAY = 60  # 초
    MAX_RETRIES = 5

    # 임베딩 api
    embedding_executor = EmbeddingExecutor(
            host=host,
            api_key=api_key,
            api_key_primary_val=api_key_primary_val
        )

    # 임베딩 벡터 차원 저장
    dimension_set = set()

    # 분할된 회의 텍스트 & 메타데이터 임베딩
    # 임베딩 결과와 메타데이터 저장
    for data in chunked_data:

        if "embedding" in data:
            dimension = len(data["embedding"])
            dimension_set.add(dimension)

        for attempt in range(MAX_RETRIES):
            try:
                # 회의 메타데이터 + 텍스트 임베딩
                request_json_string = json.dumps({
                    "text": f"회의 제목: {data['title']}, 회의 날짜: {data['date']}, 회의 참석자 수: {data['num_speakers']}, 회의 내용: {data['text']}" 
                }, ensure_ascii=False)

                request_data = json.loads(request_json_string, strict=False)
                print(request_data)
                embedding = embedding_executor.execute(request_data, request_id=request_emb) # 임베딩 실행

                data["embedding"] = embedding 

                title_list = [data['title']]
                date_list = [data['date']]
                num_speakers_list = [data['num_speakers']]
                text_list = [data['text']] # 분할된 텍스트 원본
                embedding_list = [data['embedding']] # 임베딩

                entities = [
                    title_list,
                    date_list,
                    num_speakers_list,
                    text_list,
                    embedding_list
                ]

                try:
                    insert_result = collection.insert(entities)
                    print("데이터 Insertion이 완료된 ID:", insert_result.primary_keys)
                except Exception as e:
                    print(f"데이터 삽입 중 오류 발생: {e}")

                break

            except Exception as e:
                error_code = str(e)
                if '429' in error_code: # too many requests
                    if attempt < MAX_RETRIES - 1:
                        logging.warning(f"Rate limit exceeded, retrying in {RETRY_DELAY} seconds...")
                        time.sleep(RETRY_DELAY)
                    else:
                        logging.error(f"Error while processing document after {MAX_RETRIES} attempts: {e}")
                else:
                    logging.error(f"Error while processing document: {e}")
                    break

    logging.info("Embeddings and documents have been successfully added to Milvus.")
