from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.model_selection import train_test_split
from io import BytesIO
from google.cloud import storage
from dotenv import load_dotenv
import os
import logging
import pandas as pd
import numpy as np
import joblib
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Starting preprocessing...")


# Data Loading:
def load_data(bucket_name, blob_name):
    try: 
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        contents = blob.download_as_bytes()
        df = pd.read_csv(BytesIO(contents))
        return df
    except Exception as e:
        logger.error(f"Error loading the data: {e}")
        raise


# Feature Identification:
def identify_features(df):
    target = 'Is_Fraud'
    categorical = df.select_dtypes(include=['object']).columns.tolist()
    numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target in numerical:
        numerical.remove(target)
    if target in categorical:
        categorical.remove(target)

    return categorical, numerical, target, df 


# Feature Engineering:
def feature_engineering(df):
    try:
        df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], dayfirst=True)
        df['day_of_week'] = df['Transaction_Date'].dt.dayofweek
        df['Transaction_Time'] = pd.to_datetime(df['Transaction_Time'], format='%H:%M:%S')
        df['hour'] = df['Transaction_Time'].dt.hour
        df["is_night"] = df["hour"].between(0, 5).astype(int)

        # Number of transactions in last N hours/days
        df['tx_count_last_1h'] = df.groupby('Customer_ID').rolling(window='1h', on='Transaction_Date')['Transaction_Amount'].count().values
        df['tx_count_last_24h'] = df.groupby('Customer_ID').rolling(window='24h', on='Transaction_Date')['Transaction_Amount'].count().values
        df['tx_count_last_7d'] = df.groupby('Customer_ID').rolling(window='7d', on='Transaction_Date')['Transaction_Amount'].count().values

        # Average amount in last 24h
        df['avg_amount_last_24h'] = df.groupby('Customer_ID').rolling(window='24h', on='Transaction_Date')['Transaction_Amount'].mean().values

        # Feature Selection:
        irrelevant_features = ["Customer_ID","Customer_Name","Transaction_ID",
            "Transaction_Date","Transaction_Time","Merchant_ID", 
            "Customer_Contact", "Customer_Email","Transaction_Currency","Transaction_Description"]
        df = df.drop(irrelevant_features, axis=1)

        logger.info("Feature Engineering and irrelevant columns dropped")
        logger.info(f"All columns after engineering: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise 


# Data Splitting:
def split_data(df, target):
    try:
        X = df.drop(target, axis=1)
        y = df[target].astype(int).to_numpy().ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error in data split: {e}")
        raise

# Data Preprocessing:
def preprocess_data(X_train, y_train, X_test, categorical, numerical, target):
    try:
        low_car_cat = [col for col in categorical if X_train[col].nunique() < 10]
        high_card_cat = [col for col in categorical if col not in low_car_cat]

        preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), low_car_cat),
                ('target', TargetEncoder(categories='auto', target_type='continuous', smooth='auto', cv=5, random_state=42), high_card_cat),
            ],
            remainder='passthrough'
        )

        X_train_processed = preprocessor.fit_transform(X_train, y_train)
        X_test_processed = preprocessor.transform(X_test)
        feature_names = preprocessor.get_feature_names_out()
        logger.info(f"Processed features: {feature_names}")
        return X_train_processed, X_test_processed, preprocessor, feature_names
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        raise


# Data Saving:
def save_data(X_train, y_train, X_test, y_test, feature_names, bucket_name, 
              train_blob='preprocessed_data/train.csv', test_blob='preprocessed_data/test.csv', preprocessor=None):
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_test = pd.DataFrame(X_test, columns=feature_names)
   

    logger.info(f"X_train shape: {X_train.shape}, y_train length: {len(y_train)}")
    logger.info(f"X_test shape: {X_test.shape}, y_test length: {len(y_test)}")
    logger.info(f"y_train has NaN: {pd.Series(y_train).isna().sum()}")
    logger.info(f"y_test has NaN: {pd.Series(y_test).isna().sum()}")
    logger.info(f"y_train unique values: {np.unique(y_train)}")


    df_train['target'] = pd.Series(y_train).reset_index(drop=True)
    df_test['target'] = pd.Series(y_test).reset_index(drop=True)


    def upload_to_gcs(buffer, bucket_name, blob_name):
        buffer.seek(0)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(buffer.getvalue(), content_type='text/csv')
        logger.info(f"Uploaded {blob_name} to GCS bucket {bucket_name}")
    train_buffer = BytesIO()
    df_train.to_csv(train_buffer, index=False)
    upload_to_gcs(train_buffer, bucket_name, train_blob)
    test_buffer = BytesIO()
    df_test.to_csv(test_buffer, index=False)
    upload_to_gcs(test_buffer, bucket_name, test_blob)
    

# Main:
if __name__ == "__main__":
    bucket_name = os.getenv("BUCKET_NAME")
    blob_name = os.getenv("blob_name")

    df = load_data(bucket_name, blob_name)
    logger.info("Data loaded")

    df = feature_engineering(df)
    logger.info("Feature Engineering Completed")

    categorical, numerical, target, df = identify_features(df)
    logger.info("Feature Identification completed")

    X_train, X_test, y_train, y_test = split_data(df, target)
    logger.info("Data Split Completed")

    X_train_processed, X_test_processed, preprocessor, feature_names = preprocess_data(X_train, y_train, X_test, categorical, numerical, target)
    logger.info("Data Preprocessing Completed")

    save_data(
    X_train_processed, y_train, X_test_processed, y_test,
    feature_names=feature_names,
    bucket_name=bucket_name
    )
    logger.info("Data saved to GCS")
    logger.info(f"Final shapes -> Train: {X_train_processed.shape}, Test: {X_test_processed.shape}")
    logger.info("Preprocessing pipeline completed successfully!")