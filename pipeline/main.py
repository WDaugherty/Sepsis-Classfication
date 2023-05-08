#This functions call the other functions in the pipeline to load in the data either locally or remotely 
#and then process it to create a dataframe that can be used for training the model

#Defines a funcrtion to call in loading the data from either local or remote depending on user input
def load_data(data_path, local=True):
    if local:
        # Load the data from the local path
        data = load_data_local(data_path)
    else:
        # Load the data from the remote path
        data = load_data_remote(data_path)

    return data
    
#Defines a function to call in the data processing functions
def process_data(data):
    # Process the data
    data = process_data(data)
    
    return data
