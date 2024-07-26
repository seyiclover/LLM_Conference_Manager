from pymilvus import connections, utility

# Milvus 서버 연결 정보
MILVUS_HOST = 'localhost'  # Milvus 서버 주소
MILVUS_PORT = '19530'      # Milvus 서버 포트
COLLECTION_NAME = 'meeting_data'  # 삭제할 컬렉션 이름

# Milvus 서버에 연결
connections.connect(
    alias="default", 
    host=MILVUS_HOST, 
    port=MILVUS_PORT
)

# 컬렉션 존재 여부 확인
if utility.has_collection(COLLECTION_NAME):
    # 컬렉션 삭제
    utility.drop_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' has been deleted.")
else:
    print(f"Collection '{COLLECTION_NAME}' does not exist.")

# 연결 종료 (선택사항)
connections.disconnect("default")
