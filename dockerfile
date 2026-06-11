FROM python:3.10-slim

WORKDIR /app

# 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY src/ ./src/
COPY run_pipeline.py .

# 데이터/결과 폴더 생성
RUN mkdir -p data/raw data/processed data/test \
             data/predictions models reports

CMD ["python", "run_pipeline.py"]