"""
preprocess.py
당뇨 재입원 예측 - 데이터 전처리 모듈
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """CSV 파일 로드"""
    logger.info(f"데이터 로드 중: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"로드 완료 - shape: {df.shape}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """환자 중복 제거 (patient_nbr 기준 첫 번째 입원만 유지)"""
    before = len(df)
    df = df.drop_duplicates(subset="patient_nbr", keep="first")
    logger.info(f"중복 제거: {before} → {len(df)}")
    return df


def replace_missing(df: pd.DataFrame) -> pd.DataFrame:
    """? 값을 NaN으로 치환 후 결측 컬럼 제거"""
    df = df.replace("?", np.nan)
    # 결측 비율 50% 이상 컬럼 제거
    threshold = len(df) * 0.5
    before_cols = df.shape[1]
    df = df.dropna(axis=1, thresh=threshold)
    logger.info(f"결측치 처리 - 컬럼 수: {before_cols} → {df.shape[1]}")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    타겟 변수 이진화
    readmitted: '<30' → 1 (30일 이내 재입원), 나머지 → 0
    """
    df["readmitted"] = (df["readmitted"] == "<30").astype(int)
    logger.info(f"타겟 분포:\n{df['readmitted'].value_counts()}")
    return df


def drop_unnecessary_cols(df: pd.DataFrame) -> pd.DataFrame:
    """불필요 컬럼 제거"""
    drop_cols = [
        "encounter_id", "patient_nbr",
        "examide", "citoglipton",     # 단일값 컬럼
        "payer_code", "medical_specialty",  # 결측 많음
    ]
    existing = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing)
    logger.info(f"불필요 컬럼 제거: {existing}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """범주형 변수 레이블 인코딩"""
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if "readmitted" in cat_cols:
        cat_cols.remove("readmitted")
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    logger.info(f"인코딩 완료: {len(cat_cols)}개 컬럼")
    return df


def run_preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    """전처리 전체 파이프라인 실행"""
    df = load_data(input_path)
    df = remove_duplicates(df)
    df = replace_missing(df)
    df = encode_target(df)
    df = drop_unnecessary_cols(df)
    df = encode_categoricals(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"전처리 완료 → 저장: {output_path}")
    return df


if __name__ == "__main__":
    run_preprocess(
        input_path="data/diabetic_data.csv",
        output_path="data/processed/diabetic_processed.csv"
    )
