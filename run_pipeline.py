"""
run_pipeline.py
당뇨 재입원 예측 - 전체 파이프라인 실행
 
파이프라인 순서도
==========================================
  [1] 전처리       preprocess.py
       │  data/raw/diabetic_data.csv
       ▼
  [2] 학습         train.py
       │  data/processed/diabetic_processed.csv
       ▼
  [3] 평가         evaluate.py
       │  models/xgb_model.pkl
       │  → reports/metrics.json
       │  → reports/confusion_matrix.png
       │  → reports/shap_summary.png
       ▼
  [4] 예측         predict.py
       │  data/processed/diabetic_processed.csv
       ▼
  [완료] data/predictions/predictions.csv
==========================================
Airflow 없이 로컬에서 테스트할 때 사용
"""
from src.preprocess import run_preprocess
from src.train import run_train
from src.evaluate import run_evaluate
from src.predict import run_predict
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 50)
    logger.info("STEP 1: 전처리")
    run_preprocess(
        input_path="data/diabetic_data.csv",
        output_path="data/processed/diabetic_processed.csv"
    )

    logger.info("=" * 50)
    logger.info("STEP 2: 학습")
    run_train(
        processed_path="data/processed/diabetic_processed.csv",
        model_path="models/xgb_model.pkl",
        test_data_dir="data/test"
    )

    logger.info("=" * 50)
    logger.info("STEP 3: 평가")
    metrics = run_evaluate(
        model_path="models/xgb_model.pkl",
        test_data_dir="data/test",
        report_dir="reports"
    )

    logger.info("=" * 50)
    logger.info("STEP 4: 예측")
    run_predict(
        model_path="models/xgb_model.pkl",
        input_path="data/processed/diabetic_processed.csv",
        output_path="data/predictions/predictions.csv"
    )

    logger.info("=" * 50)
    logger.info(f"파이프라인 완료. 최종 AUC: {metrics['AUC']}")


if __name__ == "__main__":
    main()
