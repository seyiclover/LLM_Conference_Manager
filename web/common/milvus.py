from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import os
import logging
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv('../backend/.env')

milvus_alias = os.getenv("MILVUS_ALIAS")
milvus_host = os.getenv("MILVUS_HOST")
milvus_port = os.getenv("MILVUS_PORT")

# milvus db 연결
def connect_to_milvus():
    try:
        connections.connect(
            alias=milvus_alias,
            host=milvus_host,
            port=milvus_port 
        )
        logging.info("Successfully connected to Milvus.")
    except Exception as e:
        logging.error(f"Error connecting to Milvus: {e}")
        raise

# 컬렉션 확인 및 생성 함수
def check_and_create_collection(collection_name):
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        print('Collection exists')

    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="num_speakers", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000), # 분할된 회의 텍스트 원본
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024) # 임베딩
        ]
        schema = CollectionSchema(fields)
        collection = Collection(name=collection_name, schema=schema, using='default', shards_num=2)

        # 인덱스 생성
        index_params = {
            "metric_type": "IP", # 거리 기준: IP(벡터 내적)
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 200}
        }
        collection.create_index(field_name="embedding", index_params=index_params)

        print('Created collection')

    collection.load()

    return collection

# Milvus 컬렉션을 가져오는 의존성 함수
def get_milvus_collection() -> Collection:
    return check_and_create_collection("meeting_data")