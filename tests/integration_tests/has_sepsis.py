import pandas as pd
import pytest
from pipeline import has_sepsis, preprocess_data

def test_integration_has_sepsis_preprocess_data():
    raw_data = pd.DataFrame({
        'row_id': [1, 2],
        'subject_id': [1, 1],
        'hadm_id': [1, 1],
        'admittime': ['2000-01-01 00:00:00', '2000-01-01 00:00:00'],
        'dischtime': ['2000-01-05 00:00:00', '2000-01-05 00:00:00'],
        'deathtime': [None, None],
        'edregtime': [None, None],
        'edouttime': [None, None],
        'diagnosis': ['TEST_DIAGNOSIS', 'TEST_DIAGNOSIS'],
        'Unnamed: 0': [0, 1],
        'DOB': ['1980-01-01', '1980-01-01'],
        'DOD': [None, None],
        'ADMITTIME': ['2000-01-01 00:00:00', '2000-01-01 00:00:00'],
        'ADMISSION_TYPE': ['EMERGENCY', 'EMERGENCY'],
        'ADMISSION_LOCATION': ['EMERGENCY ROOM', 'EMERGENCY ROOM'],
        'DISCHARGE_LOCATION': ['HOME', 'HOME'],
        'INSURANCE': ['PRIVATE', 'PRIVATE'],
        'ICD9_CODE': ['4019', '99591']
    })

    preprocessed_data = preprocess_data(raw_data)

    # Assert SEPSIS column is correctly one-hot encoded
    assert preprocessed_data['SEPSIS_1'][0] == 1
    assert preprocessed_data['SEPSIS_1'][1] == 1
 