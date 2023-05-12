import requests
from bs4 import BeautifulSoup
import os
import random
import subprocess

def get_pid_list(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        pids = [a["href"].rstrip("/") for a in soup.find_all("a") if a["href"].endswith("/")]
        return pids
    else:
        return None

def random_sample_list(input_list, sample_size):
    if sample_size >= len(input_list):
        return input_list  # Return the entire list if sample size is equal to or larger than the list length
    else:
        sampled_list = random.sample(input_list, sample_size)
        return sampled_list
    
def separate_by_prefix(common_elements):
    prefix_lists = {}
    for pid in common_elements:
        prefix = pid[:3]
        if prefix not in prefix_lists:
            prefix_lists[prefix] = []
        prefix_lists[prefix].append(pid)
    return prefix_lists

def download_data(prefix, pids):
    for pid in pids:
        url = f"https://archive.physionet.org/physiobank/database/mimic3wdb/matched/{prefix}/{pid}/"
        print(f"Downloading data for pid: {pid}")
        
        # Create the directory if it does not exist
        os.makedirs("pipeline/data/matched_wave", exist_ok=True)

        # Change the current working directory to the specified directory
        os.chdir("pipeline/data/matched_wave")

        # Run the wget command with the necessary options
        result = subprocess.run(['wget', '-r', '-N', '-c', '-np', url])

        # Change the working directory back to the original directory
        os.chdir("../../..")
        
        if result.returncode == 0:
            print(f"Successfully downloaded data for pid: {pid}")
        else:
            print(f"Failed to download data for pid: {pid}")



def filter_not_null(df, column_name):
    filtered_df = df[df[column_name].notnull()]
    return filtered_df
