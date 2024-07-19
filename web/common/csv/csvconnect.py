import os
import mysql.connector
from mysql.connector import Error
import pandas as pd
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

# CSV 파일 경로
file_path = '/mnt/a/common/meeting_data_without_content_column.csv'

try:
    # MySQL 데이터베이스에 연결
    connection = mysql.connector.connect(**connection_config)
    if connection.is_connected():
        cursor = connection.cursor()

        # CSV 파일 읽기
        df = pd.read_csv(file_path)

        # 데이터 삽입 쿼리
        insert_query = """
        INSERT INTO files (filename, speaker_count, meeting_date, file_path, user_id, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        # CSV 파일의 데이터에서 2번부터 20번까지 삽입
        for index, row in df.iloc[1:20].iterrows():
            cursor.execute(insert_query, (
                row['title'],  # filename
                row['num_speakers'],  # speaker_count
                datetime.strptime(str(row['date']), '%Y%m%d').date(),  # meeting_date
                '/mnt/a/common/uploaded_file/fix/',  # file_path
                2,  # user_id
                datetime.now(KST)  # created_at
            ))

        # 변경사항 커밋
        connection.commit()

except Error as e:
    print(f"Error: {e}")
finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
