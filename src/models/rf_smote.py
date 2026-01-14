from google.cloud import aiplatform as vertex_ai
from google.cloud import storage
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
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
default_model_path = f"gs://{os.getenv('BUCKET_NAME')}/models/fraud_rf_{timestamp}.joblib"



parser = argparse.ArgumentParser(description='Train Random Forest model')
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

    logger.info(f"Train: {X_train.shape}, Fraud rate: {y_train.mean():.4f}")
    logger.info(f"Test:  {X_test.shape}, Fraud rate: {y_test.mean():.4f}")

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train, args):
    logger.info(f"Applying SMOTE to balance the dataset:")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    logger.info(f"After SMOTE -> Train shape: {X_train_res.shape}, Fraud rate: {y_train_res.mean():.4f}")
    logger.info("Training Random Forest model on balanced data ")

    model = RandomForestClassifier(
        n_estimators=500,             
        max_depth=6,                
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        criterion='gini',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    feature_names = X_train.columns
    top_features = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)
    logger.info("Top 10 features:\n" + top_features.to_string())

    logger.info("Making predictions on training data during training phase")
    preds = model.predict(X_train)
    probs = model.predict_proba(X_train)[:, 1]

    print("\n=== PREDICTIONS ON TRAINING DATA DURING TRAINING ===")
    print("Classification Report:")
    print(classification_report(y_train, preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_train, preds))

    print(f"\nROC AUC Score on Training Data: {roc_auc_score(y_train, probs):.4f}")
    return model

def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model on test data")
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    print("\n=== Classification Report  ===")
    print(classification_report(y_test, preds))
    print("\n=== Confusion Matrix  ===")
    print(confusion_matrix(y_test, preds))
    print(f"ROC AUC Score: {roc_auc_score(y_test, probs):.4f}")

def save_model(model, output_path):
    logger.info(f"Saving model to {output_path}")
    local_path = 'random_forest_model.joblib'
    joblib.dump(model, local_path)

    if output_path.startswith('gs://'):
        import urllib.parse
        parsed = urllib.parse.urlparse(output_path)
        bucket_name = parsed.netloc
        blob_name = parsed.path.lstrip('/')

        if not blob_name or blob_name.endswith('/'):
            blob_name = blob_name + 'random_forest_model.joblib'

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)

        full_gcs_path = f"gs://{bucket_name}/{blob_name}"
        logger.info(f"Model uploaded to GCS {full_gcs_path} successfully")
    else:
        joblib.dump(model, output_path)
        logger.info(f"Model saved locally")



if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data(args.train_data, args.test_data)
    
    model = train_model(
        X_train, y_train, 
        args = args
    )
    
    evaluate_model(model, X_test, y_test)
    
    save_model(model, args.model_output)
    logger.info("Training completed successfully !")


