from google.cloud import storage
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # Important: imbalanced-learn Pipeline
import joblib
import logging
import argparse
import pandas as pd
import datetime
import urllib.parse
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dynamic model path
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
default_model_path = f"gs://{os.getenv('BUCKET_NAME', 'fraud_data_pg')}/models/fraud_rf_tuned_{timestamp}.joblib"

parser = argparse.ArgumentParser(description='Train Tuned Random Forest')
parser.add_argument('--train_data', type=str, required=True)
parser.add_argument('--test_data', type=str, required=True)
parser.add_argument('--model_output', type=str, default=default_model_path)
args = parser.parse_args()

def load_data(train_path, test_path):
    logger.info("Loading data from GCS")
    train_df = pd.read_csv(train_path).dropna(subset=['target'])
    test_df = pd.read_csv(test_path).dropna(subset=['target'])

    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    logger.info(f"Train: {X_train.shape}, Fraud rate: {y_train.mean():.4f}")
    logger.info(f"Test:  {X_test.shape}, Fraud rate: {y_test.mean():.4f}")
    return X_train, y_train, X_test, y_test

def tune_and_train_model(X_train, y_train):
    logger.info(f"Applying SMOTE to balance the dataset:")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    logger.info(f"After SMOTE -> Train shape: {X_train_res.shape}, Fraud rate: {y_train_res.mean():.4f}")
    logger.info("Training Random Forest model on balanced data ")
    
    logger.info("Starting hyperparameter tuning for Random Forest")

    param_grid = param_grid = {
        'n_estimators': [100, 200],           
        'max_depth': [10, 20, None],          
        'max_features': ['sqrt', 'log2'],     
        'min_samples_split': [2, 10],         
        'min_samples_leaf': [1, 4]            
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    logging.info("Training Random Forest model")
    base_model = RandomForestClassifier(  
        class_weight = 'balanced',
        random_state=42,
        n_jobs=-1,
        )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    grid_search.fit(X_train_res, y_train_res)
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV ROC AUC: {grid_search.best_score_:.4f}")
    best_model = grid_search.best_estimator_
    importances = best_model.feature_importances_
    feature_names = X_train_res.columns
    top_features = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)
    logger.info("Top 10 features:\n" + top_features.to_string())

    logger.info("Making predictions on training data during training phase")
    train_preds = best_model.predict(X_train_res)
    train_probs = best_model.predict_proba(X_train_res)[:, 1]

    print("\n=== PREDICTIONS ON TRAINING DATA DURING TRAINING ===")
    print("Classification Report:")
    print(classification_report(y_train_res, train_preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_train_res, train_preds))

    print(f"\nROC AUC Score on Training Data: {roc_auc_score(y_train_res, train_probs):.4f}")
    return best_model

def evaluate_model(best_model, X_test, y_test):
    logger.info("Final evaluation on hold-out test set")
 
    test_probs = best_model.predict_proba(X_test)[:, 1]
    logger.info("Default threshold is 0.5")
    test_preds = best_model.predict(X_test)
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, test_preds))
    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(y_test, test_preds))
    logger.info(f"Test ROC AUC: {roc_auc_score(y_test, test_probs):.4f}")

    print("\n Threshold tuning for fraud recall")
    thresholds = [0.1, 0.2, 0.3, 0.4]
    for thresh in thresholds:
        preds_thresh = (test_probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds_thresh).ravel()
        recall = tp/(tp+fp) if (tp+fp) >0 else 0
        precision = tp/(tp+fn) if (tp+fn) >0 else 0
        f1 = 2 * (recall) * (precision) / ((recall) + (precision))
        print(f"Thresh {thresh:.2f} | TP: {tp:4d} | FP: {fp:5d} | Recall: {recall:.3f} | Precision: {precision:.3f} | F1: {f1:.3f}")

    print("\n=== FINAL TEST SET RESULTS ===")
    print(classification_report(y_test, test_preds))
    print(f"Test ROC AUC: {roc_auc_score(y_test, test_probs):.4f}")

def save_model(best_model, output_path):
    local_path = "rf_tuned_model.joblib"
    joblib.dump(best_model, local_path)
    logger.info(f"Model saved locally: {local_path}")

    if output_path.startswith("gs://"):
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
    try:
        X_train, y_train, X_test, y_test = load_data(args.train_data, args.test_data)
        best_model = tune_and_train_model(X_train, y_train)
        evaluate_model(best_model, X_test, y_test)
        save_model(best_model, args.model_output)
        logger.info("Hyperparameter-tuned training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise