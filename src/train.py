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
    """XGBoost 학습 (클래스 불균형 처리 포함)

    원본 분석(diabetes-readmission-prediction)과 동일하게
    scale_pos_weight로 클래스 불균형을 처리한다.
    scale = 비재입원 수 / 재입원 수
    RandomizedSearchCV 튜닝 결과(learning_rate=0.01, max_depth=7,
    subsample=0.6, colsample_bytree=0.7, min_child_weight=20)를 그대로 사용한다.
    """
    scale = y_train.value_counts()[0] / y_train.value_counts()[1]
    logger.info(f"scale_pos_weight: {scale:.4f}")

    model = XGBClassifier(
        n_estimators=1000,
        early_stopping_rounds=50,
        learning_rate=0.01,
        max_depth=7,
        subsample=0.6,
        colsample_bytree=0.7,
        min_child_weight=20,
        scale_pos_weight=scale,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )
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