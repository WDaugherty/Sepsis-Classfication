import os
import pandas as pd


def load_local():
# Define file paths
    data_path = '/Users/wdaugherty/Desktop/MIMIC-III Full'
    output_path = 'pipeline/data/full_data'

    # Define the file names for admissions, patients, and diagnosis CSV.gz files
    admissions_file = 'ADMISSIONS.csv (1).gz'
    patients_file = 'PATIENTS.csv (1).gz'
    diagnosis_file = 'D_ICD_DIAGNOSES.csv (1).gz'


    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load data from local files
    admissions_data = pd.read_csv(os.path.join(data_path, admissions_file), compression='gzip')
    patients_data = pd.read_csv(os.path.join(data_path, patients_file), compression='gzip')
    diagnosis_data = pd.read_csv(os.path.join(data_path, diagnosis_file), compression='gzip')

    # Save data to output directory
    admissions_output_file_path = os.path.join(output_path, 'ADMISSIONS.csv')
    patients_output_file_path = os.path.join(output_path, 'PATIENTS.csv')
    diagnosis_output_file_path = os.path.join(output_path, 'D_ICD_DIAGNOSES.csv')

    admissions_data.to_csv(admissions_output_file_path, index=False)
    patients_data.to_csv(patients_output_file_path, index=False)
    diagnosis_data.to_csv(diagnosis_output_file_path, index=False)
