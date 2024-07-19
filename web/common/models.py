from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, relationship
from sqlalchemy import Column, Integer, String, DateTime,Date, create_engine, ForeignKey, Text, CheckConstraint
from zoneinfo import ZoneInfo
from datetime import datetime
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv('../backend/.env')

# 데이터베이스 URL 설정
DB_URL = f"mysql+pymysql://{os.getenv('USERNAME')}:{os.getenv('PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('PORT')}/{os.getenv('DBNAME')}"
# 한국 시간(KST) 설정
KST = ZoneInfo("Asia/Seoul")
# Base 클래스 생성
Base = declarative_base()

class engineconn:

    def __init__(self):
        self.engine = create_engine(DB_URL, pool_recycle=500)

    def sessionmaker(self):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        return session

    def connection(self):
        conn = self.engine.connect()
        return conn

# 데이터베이스 엔진 및 세션 설정
engine = create_engine(DB_URL, pool_recycle=500)
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

# User 모델 정의
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    google_id = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    image = Column(String(500))
    files = relationship("File", back_populates="user")  # User와 File 간의 관계 설정
    created_at = Column(DateTime, default=lambda: datetime.now(KST))

# File 모델 정의
class File(Base):
    __tablename__ = 'files'
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(500))
    speaker_count = Column(Integer, nullable=False)  # 발화자 수
    meeting_date = Column(Date, nullable=False)
    file_path = Column(String(500))
    user_id = Column(Integer, ForeignKey('users.id'))  # User 모델과의 외래키 설정
    user = relationship("User", back_populates="files")  # User와 File 간의 관계 설정
    created_at = Column(DateTime, default=lambda: datetime.now(KST))

# Transcript 모델 정의
class Transcript(Base):
    __tablename__ = 'transcripts'
    __table_args__ = (
        CheckConstraint('num_speakers >= 1 AND num_speakers <= 5'),
        {'extend_existing': True}
    )
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(Integer, ForeignKey('files.id'))
    title = Column(String(255))
    date = Column(Date, nullable=False)
    num_speakers = Column(Integer, CheckConstraint('num_speakers >= 1 AND num_speakers <= 5'))
    content = Column(Text)
    file = relationship("File")
    summary = relationship("Summary", uselist=False, back_populates="transcript")  # Transcript와 Summary 간의 관계 설정
    created_at = Column(DateTime, default=lambda: datetime.now(KST))

# Summary 모델 정의
class Summary(Base):
    __tablename__ = 'summaries'
    id = Column(Integer, primary_key=True, index=True)
    transcript_id = Column(Integer, ForeignKey('transcripts.id'), nullable=False)
    summary_text = Column(Text, nullable=False)
    transcript = relationship("Transcript", back_populates="summary")
    created_at = Column(DateTime, default=lambda: datetime.now(KST))

    @staticmethod
    def get_summary_by_transcript_id(session, transcript_id):
        return session.query(Summary).filter_by(transcript_id=transcript_id).first()

# 데이터베이스 초기화 함수
def init_db():
    Base.metadata.create_all(engine)

# get_db 함수 정의
def get_db():
    db = Session()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    init_db()