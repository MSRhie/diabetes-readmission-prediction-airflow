"""
predict.py
당뇨 재입원 예측 - 신규 데이터 예측 모듈
"""
import pandas as pd
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict(model, X: pd.DataFrame) -> pd.DataFrame:
    """예측 수행 - 확률값과 이진 예측 반환"""
    proba = model.predict_proba(X)[:, 1]
    pred  = model.predict(X)
    result = X.copy()
    result["readmit_probability"] = proba.round(4)
    result["readmit_predicted"]   = pred
    return result


def run_predict(model_path: str, input_path: str, output_path: str):
    """예측 전체 파이프라인 실행"""
    model = load_model(model_path)
    X = pd.read_csv(input_path)

    # 타겟 컬럼이 포함된 경우 제거
    if "readmitted" in X.columns:
        X = X.drop(columns=["readmitted"])

    result = predict(model, X)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info(f"예측 완료 → {output_path} (총 {len(result)}건)")
    logger.info(f"재입원 예측 건수: {result['readmit_predicted'].sum()}건 "
                f"({result['readmit_predicted'].mean()*100:.1f}%)")
    return result


if __name__ == "__main__":
    run_predict(
        model_path="models/xgb_model.pkl",
        input_path="data/processed/diabetic_processed.csv",
        output_path="data/predictions/predictions.csv"
    )
