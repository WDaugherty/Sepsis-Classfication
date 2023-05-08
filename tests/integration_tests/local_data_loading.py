import os
import shutil
import pandas as pd
import gzip
import pytest

# Define fixture for creating temporary directories
@pytest.fixture
def data_paths(tmp_path):
    data_dir = tmp_path / "data"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"

    # Create the input directory
    input_dir.mkdir(parents=True)

    # Create and save sample CSV.gz files
    sample_data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    for file_name in ["admissions.csv.gz", "patients.csv.gz", "diagnosis.csv.gz"]:
        with gzip.open(input_dir / file_name, "wt") as f:
            sample_data.to_csv(f, index=False)

    return {"input": input_dir, "output": output_dir}


# Test that the output directory is created and CSV files are loaded and saved correctly
def test_output_directory_created_and_csv_loaded(data_paths):
    input_dir = data_paths["input"]
    output_dir = data_paths["output"]

    # Check that the output directory does not exist
    assert not os.path.exists(output_dir)

    # Run the load_and_save_csv function from your script
    load_and_save_csv(input_dir, output_dir)

    # Check that the output directory now exists
    assert os.path.exists(output_dir)

    # Check that the CSV files are saved in the output directory
    assert os.path.exists(output_dir / "ADMISSIONS.csv")
    assert os.path.exists(output_dir / "PATIENTS.csv")
    assert os.path.exists(output_dir / "D_ICD_DIAGNOSES.csv")

    # Check that the content of the saved CSV files is correct
    admissions_data = pd.read_csv(output_dir / "ADMISSIONS.csv")
    patients_data = pd.read_csv(output_dir / "PATIENTS.csv")
    diagnosis_data = pd.read_csv(output_dir / "D_ICD_DIAGNOSES.csv")

    assert admissions_data.equals(pd.DataFrame({"A": [1, 2], "B": [3, 4]}))
    assert patients_data.equals(pd.DataFrame({"A": [1, 2], "B": [3, 4]}))
    assert diagnosis_data.equals(pd.DataFrame({"A": [1, 2], "B": [3, 4]}))
