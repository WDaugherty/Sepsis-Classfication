#Imports the necessary libraries
import os
import pandas as pd
from local_load_data import load_local
from remote_load_data import load_remote
from sepsis_labels import has_sepsis
from process_data import process_header_files
from matching import *

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
    diagnoses_icd = pd.read_csv(os.path.join(data_path, "DIAGNOSES_ICD.csv"))

    # Join 'admissions' and 'patients' DataFrames on 'SUBJECT_ID' column, keeping duplicates
    merged_df = pd.merge(admissions, patients, on='SUBJECT_ID', how='left')

    # Join 'merged_df' and 'diagnosis' DataFrame on 'SUBJECT_ID' column, keeping duplicates
    merged_df  = pd.merge(merged_df, diagnoses_icd, on='SUBJECT_ID', how='left')

    # Sepsis codes to consider
    sepsis_codes = ['99591', '99592', '77181','67020','67022','67024','67030','67032','67034']

    #Calls the has_sepsis function from sepsis_labels.py
    merged_df = has_sepsis(merged_df, sepsis_codes)

    #Creates a new dataset that only contains rows with sepsis
    sepsis_rows = merged_df[merged_df['has_sepsis'] == 1]

    # Extract the 'SUBJECT_ID' column as a list
    subject_id_list = sepsis_rows['SUBJECT_ID'].tolist()
    sorted_list = sorted(subject_id_list, reverse=True)

    #Creates list for pids
    pid_list = []

    #Iterates through the sorted list of subject ids
    for i in range(1, 10):
        pid_prefix = f"p{i:02d}"
        url = f"https://archive.physionet.org/physiobank/database/mimic3wdb/matched/{pid_prefix}/"
        pids = get_pid_list(url)
        if pids:
            pid_list.extend(pids)


    # Remove the first element and the last two elements from the pid_list
    if pid_list:
        pid_list = pid_list[1:-2]

    #Convert subject_ids to waveform directory ids
    converted_list = ['p{:06d}'.format(int(pid)) for pid in sorted_list]

    #Defines the sample size
    sample_size = 100

    #Calls the random_sample_list function from matching.py
    converted_list = random_sample_list(converted_list, sample_size)

    #Checks for common elements between the pid_list and the converted_list
    common_elements = list(set(pid_list) & set(converted_list))

    prefix_lists = separate_by_prefix(common_elements)


    for prefix, pids in prefix_lists.items():
        download_data(prefix, pids)

    # Initialize an empty dataframe
    master_df = pd.DataFrame()

    # Set the base directory path
    base_dir = '/pipeline/data/matched_wave/'

    # Iterate through all the folders
    folder_dfs = []
    for root, dirs, files in os.walk(base_dir):
        # Check if there are header files in the current folder
        header_files = [f for f in files if f.endswith('.hea')]
        
        if header_files:
            # Read and process the header files to create dataframes
            folder_df = process_header_files(header_files, root)
            
            # Append the folder dataframe to the folder_dfs list
            folder_dfs.append(folder_df)

    # Concatenate all the folder dataframes into the master dataframe
    master_df = pd.concat(folder_dfs, ignore_index=True)

    master_df = sepsis_rows.join(master_df, on='SUBJECT_ID', how='left' , rsuffix='left')

    # Call the function to filter the DataFrame based on a specific column
    filtered_df = filter_not_null(master_df, 'signal_length')

    filtered_df.to_csv('/pipeline/data/processed_data/filter_df.csv', index=False)

if __name__ == '__main__': 
    main()




