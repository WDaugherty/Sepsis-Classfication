#Import necessary libraries
import os
import pytest

#Define fixture for creating temporary directory
@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / 'data'

#Test that the data directory is created
def test_data_directory_created(data_dir):
# Check that the data directory does not exist
    assert not os.path.exists(data_dir)

    # Create the data directory
    data_dir.mkdir()

    # Check that the data directory now exists
    assert os.path.exists(data_dir)