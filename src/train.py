"""
train.py
당뇨 재입원 예측 - 모델 학습 모듈
"""
import pandas as pd
import numpy as np
import pickle
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET = "readmitted"


def load_processed(filepath: str) -> pd.DataFrame:
    logger.info(f"전처리 데이터 로드: {filepath}")
    return pd.read_csv(filepath)


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """학습/검증 데이터 분리"""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train) -> XGBClassifier:
    """XGBoost 학습 (클래스 불균형 처리 포함)"""
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    logger.info("모델 학습 완료")
    return model


def save_model(model, model_path: str):
    """학습된 모델 저장"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"모델 저장: {model_path}")


def save_test_data(X_test, y_test, output_dir: str):
    """검증 데이터 저장 (evaluate.py에서 사용)"""
    os.makedirs(output_dir, exist_ok=True)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    logger.info(f"검증 데이터 저장: {output_dir}")


def run_train(processed_path: str, model_path: str, test_data_dir: str):
    """학습 전체 파이프라인 실행"""
    df = load_processed(processed_path)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    save_model(model, model_path)
    save_test_data(X_test, y_test, test_data_dir)
    return model


if __name__ == "__main__":
    run_train(
        processed_path="data/processed/diabetic_processed.csv",
        model_path="models/xgb_model.pkl",
        test_data_dir="data/test"
    )
