import pandas as pd
import numpy as np

def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(['Unnamed: 0', 'ROW_ID_x', 'ROW_ID_y', 'ROW_ID', 'HADM_ID_y', 'SUBJECT_IDleft', 'ICD9_CODE'], axis=1, errors='ignore')

    # Convert date columns to datetime
    date_columns = ['ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'EDREGTIME', 'EDOUTTIME', 'DOB', 'DOD', 'DOD_HOSP', 'DOD_SSN']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])

    # Convert datetime columns to numeric (e.g., number of days since a reference date)
    ref_date = pd.Timestamp('2000-01-01')
    for col in date_columns:
        df[col] = (df[col] - ref_date).dt.total_seconds() / (24 * 60 * 60)

    # Fill missing values with -1 (assuming all time-related columns are positive)
    df[date_columns] = df[date_columns].fillna(-1)

    # One-hot encode categorical columns
    cat_columns = ['ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'DIAGNOSIS', 'GENDER']
    df = pd.get_dummies(df, columns=cat_columns, dummy_na=True)

    # Fill missing values for the 'SEQ_NUM' column with -1
    df['SEQ_NUM'] = df['SEQ_NUM'].fillna(-1)

    # Remove any remaining columns with non-numeric data types
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    df = df.drop(non_numeric_columns, axis=1)

    # Ensure all finite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df

def sample_data(df, sample_size, random_state=None):
    """
    Randomly sample the dataframe.
    """
    return df.sample(sample_size, random_state=random_state)