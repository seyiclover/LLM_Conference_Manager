import os
import json
import http.client
import logging
from dotenv import load_dotenv
from urllib.parse import urlparse
from fastapi import HTTPException
from http import HTTPStatus
from urllib.parse import urlparse

# .env 파일에서 환경 변수 로드
load_dotenv(dotenv_path='../backend/.env')

# 환경 변수 사용
host_url = os.getenv("HOST")
api_key = os.getenv("API_KEY")
api_key_primary_val = os.getenv("API_KEY_PRIMARY_VAL")
request_sum = os.getenv("REQUEST_SUM")
request_chat = os.getenv("REQUEST_CHAT")

# 호스트 추출
parsed_url = urlparse(host_url)
host = parsed_url.hostname

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class STT_CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_sum):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_sum

    def _send_request(self, completion_request):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }
        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', '/testapp/v1/api-tools/summarization/v2/e8d81bee36194f4b8bbbd374584c30fa', json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result

    def execute(self, completion_request):
        res = self._send_request(completion_request)
        if res['status']['code'] == '20000':
            return res['result']['text']
        else:
            logger.error(f"Error in summarization API: {res}")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail='Summarization API error')

class CLOVAStudioExecutor:
    def __init__(self, host, api_key, api_key_primary_val):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val

    def _send_request(self, completion_request, endpoint, request_id):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': request_id
        }
        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', endpoint, json.dumps(completion_request), headers)
        response = conn.getresponse()
        status = response.status
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result, status

class ChatCompletionExecutor(CLOVAStudioExecutor):
    def __init__(self, host, api_key, api_key_primary_val):
        super().__init__(host, api_key, api_key_primary_val)

    def execute(self, completion_request, endpoint='/testapp/v1/chat-completions/HCX-003', stream=False, request_id=None):
        res, status = self._send_request(completion_request, endpoint, request_id)
        if status == HTTPStatus.OK:
            return res
        else:
            raise ValueError(f"Error: HTTP {status}, message: {res.get('message', 'Unknown error')}")

class SummarizationExecutor(CLOVAStudioExecutor):
    def __init__(self, host, api_key, api_key_primary_val):
        super().__init__(host, api_key, api_key_primary_val)

    def execute(self, completion_request, endpoint='/testapp/v1/api-tools/summarization/v2/e8d81bee36194f4b8bbbd374584c30fa', request_id=None):
        res, status = self._send_request(completion_request, endpoint, request_id)
        if status == HTTPStatus.OK and "result" in res:
            return res["result"]["text"]
        else:
            error_message = res.get("status", {}).get("message", "Unknown error") if isinstance(res, dict) else "Unknown error"
            raise ValueError(f"Error: HTTP {status}, message: {error_message}")

class SegmentationExecutor(CLOVAStudioExecutor):
    def __init__(self, host, api_key, api_key_primary_val):
        super().__init__(host, api_key, api_key_primary_val)

    def execute(self, completion_request, endpoint='/testapp/v1/api-tools/segmentation/c579867d7bd84cb1a53f6127beae3805', request_id=None):
        res, status = self._send_request(completion_request, endpoint, request_id)
        if status == HTTPStatus.OK and "result" in res:
            return res["result"]['topicSeg']
        else:
            error_message = res.get("status", {}).get("message", "Unknown error") if isinstance(res, dict) else "Unknown error"
            raise ValueError(f"Error: HTTP {status}, message: {error_message}")

class EmbeddingExecutor(CLOVAStudioExecutor):
    def __init__(self, host, api_key, api_key_primary_val):
        super().__init__(host, api_key, api_key_primary_val)

    def execute(self, completion_request, endpoint='/testapp/v1/api-tools/embedding/clir-emb-dolphin/642db2c971e6452882c1a8421821866c', request_id=None):
        res, status = self._send_request(completion_request, endpoint, request_id)
        if status == HTTPStatus.OK and "result" in res:
            return res["result"]["embedding"]
        else:
            error_message = res.get("status", {}).get("message", "Unknown error") if isinstance(res, dict) else "Unknown error"
            raise ValueError(f"Error: HTTP {status}, message: {error_message}")