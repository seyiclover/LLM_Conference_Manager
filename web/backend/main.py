import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
# .env 파일 로드
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# 프로젝트의 루트 디렉토리를 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


from common.models import init_db
from backend.auth.routes import router as auth_router
from backend.users.routes import router as users_router
from backend.upload.routes import router as upload_router
from backend.clova.routes import router as clova_router
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(lifespan=lifespan)

# 서버 시작 메시지
print("welcome server on")

static_dir = "/Users/jhpark/Documents/study/ai/projects/nexochat/embedding_rag/LLM_Conference_Manager/web/front/build/static"
index_file_path = "/Users/jhpark/Documents/study/ai/projects/nexochat/embedding_rag/LLM_Conference_Manager/web/front/build/index.html"

# 정적 파일 제공 설정
app.mount("/static", StaticFiles(directory=static_dir), name="static")



# 데이터베이스 초기화
init_db()

# 라우터 등록
app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(users_router, prefix="/users", tags=["users"])
app.include_router(upload_router, prefix="/upload", tags=["upload"])
app.include_router(clova_router, prefix="/clova", tags=["clova"])


# CORS 설정 추가
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://localhost:7000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 인덱스 파일 제공 설정
@app.get("/")
async def read_index():
    print(f"Serving index file from: {index_file_path}")
    if os.path.exists(index_file_path):
        return FileResponse(index_file_path)
    return HTMLResponse(content="Index file not found", status_code=404)

@app.get("/{catchall:path}")
async def read_react_routes(catchall: str):
    print(f"Serving react route: {catchall}")
    if os.path.exists(index_file_path):
        return FileResponse(index_file_path)
    return HTMLResponse(content="Index file not found", status_code=404)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
