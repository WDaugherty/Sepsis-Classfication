import os
import pandas as pd
import re
import pipeline
from pipeline.data.process_data import process_header_files
import sys
sys.path.append("pipeline")


# A sample directory containing header files for testing
test_dir = "/pipeline/data_preprocessing/data/raw_data/archive.physionet.org/physiobank/database/mimic3wdb/matched"

def test_process_header_files_unit():
    # A list of header files to process
    header_files = ["test_1.hea", "test_2.hea"]
    
    # Call the process_header_files function
    result = process_header_files(header_files, test_dir)

    # Check that the result is a pandas dataframe
    assert isinstance(result, pd.DataFrame)
    
    # Check that the dataframe has the expected columns
    expected_columns = ["subject_id", "record_name", "signal_length"]
    assert all(col in result.columns for col in expected_columns)

def test_process_header_files_integration():
    # Call the process_header_files function on the entire directory
    result = process_header_files(os.listdir(test_dir), test_dir)

    # Check that the result is a pandas dataframe
    assert isinstance(result, pd.DataFrame)

    # Check that the dataframe has the expected columns
    expected_columns = ["subject_id", "record_name", "signal_length"]
    assert all(col in result.columns for col in expected_columns)

    # Check that the dataframe has the expected number of rows
    expected_rows = 2  # Change this to the expected number of rows
    assert len(result) == expected_rows

def test_process_data():
    # Load the merged dataframe
    merged_df = pd.read_csv('pipeline/data_preprocessing/data/processed_data/merged_data.csv')

    # Check that the dataframe has the expected columns
    expected_columns = ['subject_id', 'hadm_id', 'admittime', 'dob', 'gender']
    assert all(col in merged_df.columns for col in expected_columns)

    # Check that the dataframe has the expected number of rows
    expected_rows = 1000  # Change this to the expected number of rows
    assert len(merged_df) == expected_rows

    # Check that the date/time columns are in datetime format
    assert isinstance(merged_df['admittime'][0], pd.Timestamp)
    assert isinstance(merged_df['dob'][0], pd.Timestamp)