import pandas as pd
import pytest
from your_module import has_sepsis, preprocess_data

def test_has_sepsis():
    assert has_sepsis(['99591', 'V3000']) == 1
    assert has_sepsis(['V3000', '78552']) == 1
    assert has_sepsis(['4019', 'V3000']) == 0
    assert has_sepsis([]) == 0

def test_preprocess_data():
    raw_data = pd.DataFrame({
        'row_id': [1],
        'subject_id': [1],
        'hadm_id': [1],
        'admittime': ['2000-01-01 00:00:00'],
        'dischtime': ['2000-01-05 00:00:00'],
        'deathtime': [None],
        'edregtime': [None],
        'edouttime': [None],
        'diagnosis': ['TEST_DIAGNOSIS'],
        'Unnamed: 0': [0],
        'DOB': ['1980-01-01'],
        'DOD': [None],
        'ADMITTIME': ['2000-01-01 00:00:00'],
        'ADMISSION_TYPE': ['EMERGENCY'],
        'ADMISSION_LOCATION': ['EMERGENCY ROOM'],
        'DISCHARGE_LOCATION': ['HOME'],
        'INSURANCE': ['PRIVATE'],
        'ICD9_CODE': ['4019']
    })

    preprocessed_data = preprocess_data(raw_data)

    # Assert columns are dropped and new columns are created
    assert set(preprocessed_data.columns) == {'age', 'ADMISSION_TYPE_EMERGENCY', 'ADMISSION_LOCATION_EMERGENCY ROOM',
                                              'DISCHARGE_LOCATION_HOME', 'INSURANCE_PRIVATE', 'SEPSIS_1'}

    # Assert age calculation
    assert preprocessed_data['age'][0] == pytest.approx(43, rel=1)

    # Assert one-hot encoding of categorical variables
    assert preprocessed_data['ADMISSION_TYPE_EMERGENCY'][0] == 1
    assert preprocessed_data['ADMISSION_LOCATION_EMERGENCY ROOM'][0] == 1
    assert preprocessed_data['DISCHARGE_LOCATION_HOME'][0] == 1
    assert preprocessed_data['INSURANCE_PRIVATE'][0] == 1

    # Assert one-hot encoding of SEPSIS column
    assert preprocessed_data['SEPSIS_1'][0] == 0