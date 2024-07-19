import os
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv('../backend/.env')

# MySQL 데이터베이스 연결 설정
connection_config = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DBNAME'),
    'user': os.getenv('USERNAME'),
    'password': os.getenv('PASSWORD'),
    'port': os.getenv('PORT')
}

# 한국 시간(KST) 설정
KST = ZoneInfo("Asia/Seoul")

try:
    # MySQL 데이터베이스에 연결
    connection = mysql.connector.connect(**connection_config)
    if connection.is_connected():
        cursor = connection.cursor()

        # file_id가 12인 레코드를 조회
        select_query = "SELECT id FROM transcripts WHERE file_id = 12"
        cursor.execute(select_query)
        records = cursor.fetchall()

        # 첫 번째 레코드는 file_id가 12로 유지되고, 그 다음부터는 순차적으로 증가
        new_file_id = 13
        for index, record in enumerate(records):
            if index == 0:
                continue
            update_query = "UPDATE transcripts SET file_id = %s WHERE id = %s"
            cursor.execute(update_query, (new_file_id, record[0]))
            new_file_id += 1

        # 변경사항 커밋
        connection.commit()

except Error as e:
    print(f"Error: {e}")
finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
