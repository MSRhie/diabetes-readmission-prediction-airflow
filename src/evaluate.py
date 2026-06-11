"""
evaluate.py
당뇨 재입원 예측 - 모델 평가 + SHAP 해석 모듈
"""
import pandas as pd
import numpy as np
import pickle
import os
import json
import logging
import matplotlib
matplotlib.use("Agg")  # 서버 환경 (GUI 없음)
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_test_data(test_data_dir: str):
    X_test = pd.read_csv(f"{test_data_dir}/X_test.csv")
    y_test = pd.read_csv(f"{test_data_dir}/y_test.csv").squeeze()
    return X_test, y_test


def compute_metrics(model, X_test, y_test) -> dict:
    """주요 성능 지표 계산"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "AUC": round(auc, 4),
        "Precision_class1": round(report["1"]["precision"], 4),
        "Recall_class1":    round(report["1"]["recall"], 4),
        "F1_class1":        round(report["1"]["f1-score"], 4),
        "Accuracy":         round(report["accuracy"], 4),
    }
    logger.info(f"성능 지표: {metrics}")
    return metrics


def save_metrics(metrics: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"지표 저장: {output_path}")


def plot_confusion_matrix(model, X_test, y_test, output_path: str):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No readmit", "Readmit <30"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"혼동행렬 저장: {output_path}")


def plot_shap_summary(model, X_test, output_path: str, max_display: int = 15):
    """SHAP summary plot 저장"""
    explainer = shap.TreeExplainer(model)
    # 속도를 위해 최대 2000개 샘플만 사용
    sample = X_test.sample(min(2000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, max_display=max_display, show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP plot 저장: {output_path}")


def run_evaluate(model_path: str, test_data_dir: str, report_dir: str):
    """평가 전체 파이프라인 실행"""
    model = load_model(model_path)
    X_test, y_test = load_test_data(test_data_dir)

    metrics = compute_metrics(model, X_test, y_test)
    save_metrics(metrics, f"{report_dir}/metrics.json")
    plot_confusion_matrix(model, X_test, y_test, f"{report_dir}/confusion_matrix.png")
    plot_shap_summary(model, X_test, f"{report_dir}/shap_summary.png")

    logger.info(f"평가 완료 - 리포트 저장 위치: {report_dir}")
    return metrics


if __name__ == "__main__":
    run_evaluate(
        model_path="models/xgb_model.pkl",
        test_data_dir="data/test",
        report_dir="reports"
    )
