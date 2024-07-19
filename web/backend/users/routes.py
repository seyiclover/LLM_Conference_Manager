from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from common.models import get_db, User as UserModel

router = APIRouter()

@router.get("/users")
def read_users(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    users = db.query(UserModel).offset(skip).limit(limit).all()
    return users
