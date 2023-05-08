#Imports the necessary libraries
import os
import pandas as pd
from local_load_data import load_local
from remote_load_data import load_remote
from sepsis_labels import preprocess_data
from process_data import process_header_files

#Make a function that asks the user if they want to load the data either locally or remotely
def load_data():
    #Ask the user if they want to load the data locally or remotely
    load_type = input("Do you want to load the data locally or remotely? (local/remote) ")
    #If the user wants to load the data locally, call the load_local function
    if load_type == "local":
        load_local()
        return "local"
    #If the user wants to load the data remotely, call the load_remote function
    elif load_type == "remote":
        load_remote()
        return "remote"
    #If the user inputs something other than local or remote, ask them to input local or remote
    else:
        print("Please input local or remote")
        return load_data()

#Create the main function that will call the other functions in the pipeline based on the result of load_data() function
def main():
    data_type = load_data()

    if data_type == "local":
        data_path = "pipeline/data/full_data/"
    elif data_type == "remote":
        data_path = "data/raw_data/"

    # Read in the files that were loaded in the load_data() function
    admissions = pd.read_csv(os.path.join(data_path, "ADMISSIONS.csv"))
    patients = pd.read_csv(os.path.join(data_path, "PATIENTS.csv"))
    diagnoses_icd = pd.read_csv(os.path.join(data_path, "D_ICD_DIAGNOSES.csv"))

    # Preprocess the data using preprocess_data() function
    admissions_df = preprocess_data(admissions)

    patients_df = preprocess_data(patients)

    icd_df = preprocess_data(diagnoses_icd)


    # Initialize an empty dataframe
    master_df = pd.DataFrame()

    # Set the base directory path
    base_dir = '/path/to/Sepsis-Classfication/project/data/raw_data/archive.physionet.org/physiobank/database/mimic3wdb/matched'

    # Iterate through all the folders
    for root, dirs, files in os.walk(base_dir):
        # Check if there are header files in the current folder
        header_files = [f for f in files if f.endswith('.hea')]

        if header_files:
            # Read and process the header files to create dataframes
            folder_df = process_header_files(header_files, root)

            # Append the folder dataframe to the master dataframe
            master_df = pd.concat([master_df, folder_df], ignore_index=True)


    # Merge the admission and patient data based on the patient ID
    merged_df = pd.merge(admissions_df, patients_df, on='subject_id')

    # Filter the merged dataset to include only the columns of interest
    merged_df = merged_df[['subject_id', 'hadm_id', 'admittime', 'dob', 'gender']]

    # Convert the date/time columns to datetime format
    merged_df['admittime'] = pd.to_datetime(merged_df['admittime'])
    merged_df['dob'] = pd.to_datetime(merged_df['dob'])

    # Save the merged dataframe to pipeline/data_preprocessing/data/processed_data
    merged_df.to_csv('pipeline/data_preprocessing/data/processed_data/merged_data.csv', index=False)

    # Save the merged dataframe to pipeline/data_preprocessing/data/processed_data
    merged_df.to_csv('pipeline/data_preprocessing/data/processed_data/merged_data.csv', index=False)

if __name__ == '__main__':
    main()




