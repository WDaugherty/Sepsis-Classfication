import pandas as pd

def has_sepsis(icd9_codes):
    """
    Check if a list of ICD-9 codes includes any codes for sepsis.
    """
    sepsis_codes = ['99591', '99592', '78552']
    return int(any(code in sepsis_codes for code in icd9_codes))

def preprocess_data(data):
    """
    Preprocess the input data by dropping unnecessary columns, converting date columns to age,
    one-hot encoding categorical variables, and one-hot encoding the SEPSIS column.
    """
    # Drop unnecessary columns
    columns_to_drop = ['row_id', 'subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime', 'diagnosis','Unnamed: 0']
    data = data.drop(columns=columns_to_drop)

    # Convert date columns to age
    date_columns = ['DOB', 'DOD', 'ADMITTIME']
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])
    data['age'] = (pd.Timestamp('now') - data['DOB']).astype('<m8[Y]')
    data = data.drop(columns=['DOB'])

    # One-hot encode categorical variables
    categorical_columns = ['ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION','INSURANCE']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # One-hot encode the SEPSIS column
    grouped_diagnoses = data.groupby(['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].apply(list).reset_index()
    grouped_diagnoses['SEPSIS'] = grouped_diagnoses['ICD9_CODE'].apply(has_sepsis)
    data_with_labels = data.merge(grouped_diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEPSIS']], on=['SUBJECT_ID', 'HADM_ID'], how='left')
    data_with_labels['SEPSIS'] = data_with_labels['SEPSIS'].fillna(0).astype(int)
    sepsis_dummies = pd.get_dummies(data_with_labels['SEPSIS'], prefix='SEPSIS', drop_first=True)
    data_with_labels = pd.concat([data_with_labels.drop(columns=['SEPSIS']), sepsis_dummies], axis=1)

    return data_with_labels