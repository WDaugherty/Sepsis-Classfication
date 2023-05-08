import os
import re
import pandas as pd
import sys

def process_header_files(header_files, folder_path):
    records = []

    for header_file in header_files:
        with open(os.path.join(folder_path, header_file), 'r') as f:
            content = f.read()

        record_name = header_file[:-4]  # Remove the '.hea' extension

        # Extract subject_id from the record_name
        subject_id = int(re.findall(r'\d+', record_name)[0])

        # Extract sample frequency and record duration
        sample_frequency_match = re.findall(r'^\S+ \d+ ([\d.]+)\/(\d+)', content, re.MULTILINE)
        sample_frequency_match_2 = re.findall(r'^\S+ \d+ (\d+)', content, re.MULTILINE)
        record_duration_match = re.findall(r'\d+', content)
        record_duration = int(record_duration_match[-1]) if record_duration_match else None

        if not sample_frequency_match and not sample_frequency_match_2:
            print(f"Cannot find sample frequency or record duration in header file {header_file}:\n{content}")
            continue

        if sample_frequency_match:
            sample_frequency = float(sample_frequency_match[0][0]) / float(sample_frequency_match[0][1])
        else:
            sample_frequency = int(sample_frequency_match_2[0])

        if not record_duration_match:
            print(f"Cannot find record duration in header file {header_file}:\n{content}")
            continue

        record_duration = int(record_duration_match[0])

        # Calculate signal length
        signal_length = int(sample_frequency * record_duration)

        records.append({
            'subject_id': subject_id,
            'record_name': record_name,
            'signal_length': signal_length
        })

    return pd.DataFrame(records)

