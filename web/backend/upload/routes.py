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

router = APIRouter()

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