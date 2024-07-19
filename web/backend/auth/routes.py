from fastapi import APIRouter, Depends, HTTPException, Request, Response, Header
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta, timezone
import os, jwt
from common.models import get_db, User as UserModel
from jose import JWTError
import logging
from dotenv import load_dotenv


router = APIRouter()

load_dotenv(dotenv_path='../.env')

# 환경 변수에서 클라이언트 ID와 JWT 비밀 키를 가져옴
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
JWT_SECRET = os.getenv("JWT_SECRET")
ALGORITHM = os.getenv("ALGORITHM")

if not GOOGLE_CLIENT_ID or not JWT_SECRET:
    raise ValueError("Missing required environment variables: GOOGLE_CLIENT_ID and JWT_SECRET")

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=1)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)
    return encoded_jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
def get_user_by_google_id(db, google_id: str):
    return db.query(UserModel).filter(UserModel.google_id == google_id).first()


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, ALGORITHM)
        google_id: str = payload.get("sub")
        if google_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user_by_google_id(db, google_id=google_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.post("/callback")
async def auth_callback(request: Request, response: Response, db: Session = Depends(get_db)):
    try:
        body = await request.json()
        google_token = body.get('token')
        if not google_token:
            raise HTTPException(status_code=400, detail="Token is missing")

        id_info = id_token.verify_oauth2_token(
            google_token, google_requests.Request(), GOOGLE_CLIENT_ID
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Token verification failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {str(e)}")

    user = db.query(UserModel).filter(UserModel.google_id == id_info["sub"]).first()
    if user is None:
        user = UserModel(
            google_id=id_info["sub"],
            email=id_info["email"],
            name=id_info["name"],
            image=id_info.get("picture")  # 프로필 이미지 URL 저장
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        # 사용자가 이미 존재하는 경우, 필요한 경우 업데이트
        user.email = id_info["email"]
        user.name = id_info["name"]
        user.image = id_info.get("picture")
        db.commit()

    token_data = {
        "sub": user.google_id,
        "email": user.email,
        "name": user.name,
        "picture": user.image  # 토큰에 프로필 이미지 URL 추가
    }
    token = create_access_token(token_data)
    print(token)

    return JSONResponse(content={"token": token})

@router.get("/secure-data")
def secure_data(authorization: str = Header(None), db: Session = Depends(get_db)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Not authenticated")

    token = authorization.split(" ")[1]

    try:
        if token.count(".") != 2:
            logging.error(f"Invalid token format: {token}")
            raise HTTPException(status_code=403, detail="Invalid token format")

        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=403, detail="Invalid token")
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=403, detail=f"Invalid token: {str(e)}")

    user = db.query(UserModel).filter(UserModel.google_id == user_id).first()
    if user is None:  # 사용자가 데이터베이스에 없는 경우 처리
        raise HTTPException(status_code=404, detail="User not found")

    return {"message": "This is secure data", "user": user}
