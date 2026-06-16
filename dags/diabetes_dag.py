"""
diabetes_dag.py
당뇨 재입원 예측 파이프라인 - Airflow DAG

흐름: 전처리 → 학습 → 평가 → 예측
"""
from datetime import datetime
from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

import sys
sys.path.insert(0, "/opt/airflow")

from src.preprocess import run_preprocess
from src.train import run_train
from src.evaluate import run_evaluate
from src.predict import run_predict


# 컨테이너 내부 기준 경로 (volumes 마운트와 일치해야 함)
RAW_PATH        = "/opt/airflow/data/raw/diabetic_data.csv"
PROCESSED_PATH  = "/opt/airflow/data/processed/diabetic_processed.csv"
MODEL_PATH      = "/opt/airflow/models/xgb_model.pkl"
TEST_DATA_DIR   = "/opt/airflow/data/test"
REPORT_DIR      = "/opt/airflow/reports"
PREDICTIONS_OUT = "/opt/airflow/data/predictions/predictions.csv"


def task_preprocess():
    run_preprocess(input_path=RAW_PATH, output_path=PROCESSED_PATH)


def task_train():
    run_train(
        processed_path=PROCESSED_PATH,
        model_path=MODEL_PATH,
        test_data_dir=TEST_DATA_DIR,
    )


def task_evaluate():
    run_evaluate(
        model_path=MODEL_PATH,
        test_data_dir=TEST_DATA_DIR,
        report_dir=REPORT_DIR,
    )


def task_predict():
    run_predict(
        model_path=MODEL_PATH,
        input_path=PROCESSED_PATH,
        output_path=PREDICTIONS_OUT,
    )


with DAG(
    dag_id="diabetes_readmission_pipeline",
    description="당뇨 재입원 예측 - 전처리/학습/평가/예측 파이프라인",
    start_date=datetime(2026, 1, 1),
    schedule=None,        # 수동 실행 (필요 시 "@daily" 등으로 변경)
    catchup=False,
    tags=["diabetes", "ml", "portfolio"],
) as dag:

    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=task_preprocess,
    )

    train = PythonOperator(
        task_id="train",
        python_callable=task_train,
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=task_evaluate,
    )

    predict = PythonOperator(
        task_id="predict",
        python_callable=task_predict,
    )

    # 실행 순서: 전처리 → 학습 → 평가 → 예측
    preprocess >> train >> evaluate >> predict
