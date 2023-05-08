#Defines the main function for the gen_models module
#Imports the necessary libraries
import numpy as np
import pandas as pd

#Import the necessary functions from the other modules
from smote_gan import smote_gan
from autoencoder import generate_autoencoder_data
#from Flow_Models import *
#from MCMc import *
#from VarEncoders import *

#Defines the main function for the gen_models module that takes in the new_mimic.csv file from pipeline/data/proccessed_data
def main():
    #Loads in the data
    df = pd.read_csv("pipeline/data/processed_data/new_mimic.csv")

    #Calls the generate smote_gan function from the smote_gan module
    smote_gan(df)

    #Calls the generate autoencoder function from the autoencoder module
    generate_autoencoder_data(df)
