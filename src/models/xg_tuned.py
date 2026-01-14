from google.cloud import aiplatform as vertex_ai
from google.cloud import storage
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, average_precision_score, precision_recall_curve
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
import joblib
import logging
import argparse
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
default_model_path = f"gs://{os.getenv('BUCKET_NAME')}/models/fraud_xgb_{timestamp}.joblib"



parser = argparse.ArgumentParser(description='Train XGBoost model')
parser.add_argument('--train_data', type=str, required=True)
parser.add_argument('--test_data', type=str, required=True)
parser.add_argument('--model_output', type=str, 
                    default=default_model_path,
                    help='GCS path to save model')
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--max_depth', type=int, default=None)
args = parser.parse_args()

def load_data(train_path, test_path):
    logging.info("Loading training and test data from GCS")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = train_df.dropna(subset=['target'])
    test_df = test_df.dropna(subset=['target'])
    
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train, args):
    logging.info("Training XGBoost model")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() 
    logger.info(f"scale_pos_weight = {scale_pos_weight:.1f}")

    logger.info("Starting hyperparameter tuning for XGBoost")

    param_grid = {
        'n_estimators': [500, 600],           
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.2],
        'min_child_weight': [1, 3],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8]          
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    logging.info("Training XGBoost model")

    base_model = XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        random_state=42)

    model = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,                  
        scoring='roc_auc',     
        n_jobs=-1,
        verbose=1
)

    model.fit(X_train, y_train)
    logger.info(f"Best parameters: {model.best_params_}")
    logger.info(f"Best CV ROC AUC: {model.best_score_:.4f}")
    best_model = model.best_estimator_

    logger.info("Making predictions on training data during training phase")
    preds = best_model.predict(X_train)
    probs = best_model.predict_proba(X_train)[:, 1]

    print("\n=== PREDICTIONS ON TRAINING DATA DURING TRAINING ===")
    print("Classification Report:")
    print(classification_report(y_train, preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_train, preds))

    print(f"\nROC AUC Score on Training Data: {roc_auc_score(y_train, probs):.4f}")
    return best_model

def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model on test data")
    probs = model.predict_proba(X_test)[:, 1]

    print("\n=== DEFAULT THRESHOLD (0.5) ===")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\n=== THRESHOLD TUNING ===")
    thresholds = [0.1, 0.2, 0.3, 0.45, 0.5]
    for thresh in thresholds:
        preds_thresh = (probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds_thresh).ravel()
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
        print(f"Thresh {thresh:.2f} | Recall: {recall:.3f} | Precision: {precision:.3f} | F1: {f1:.3f} | TP: {tp} | FP: {fp}")

    print(f"\nROC AUC: {roc_auc_score(y_test, probs):.4f}")
    print(f"PR AUC (Average Precision): {average_precision_score(y_test, probs):.4f}")

def save_model(best_model, output_path):
    local_path = "model.joblib"
    joblib.dump(best_model, local_path)
    logger.info(f"Model saved locally: {local_path}")

    if output_path.startswith("gs://"):
        import urllib.parse
        parsed = urllib.parse.urlparse(output_path)
        bucket_name = parsed.netloc
        blob_name = parsed.path.lstrip('/') or local_path

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)

        full_path = f"gs://{bucket_name}/{blob_name}"
        logger.info(f"Model uploaded: {full_path}")
        print(f"MODEL_ARTIFACT_URI={full_path}")
    else:
        joblib.dump(best_model, output_path)


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data(args.train_data, args.test_data)
    
    model = train_model(
        X_train, y_train,
        args
    )
    
    evaluate_model(model, X_test, y_test)
    
    save_model(model, args.model_output)
    logger.info("Training completed successfully !")


