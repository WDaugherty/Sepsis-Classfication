#This file will take in the necessary libaries and functions from the other modules and run them such that we can generate teh data
#that we need for the models, evlaute the models, transfer learning, and then evaluate the transfer learning models and then 
#apply an lstm model to the final output. 

#Calls the necessary libraries
import numpy as np
import pandas as pd
import os
import sys
import re


#Calls the subfucntions from the other modules
from gen_models.smote_gan import smote_gan
from gen_models.autoencoder import generate_autoencoder_data
from gen_models.Flow_Models import *
from gen_models.MCMc import *
from gen_models.VarEncoders import *
#from LSTM import * NOT DONE YET
from analysis.comparative_gen import *
from analysis.ground_truth_mimic import *

#Defines the main function for the gen_models module that takes in the new_mimic.csv file from pipeline/data/proccessed_data
def main():
    #Loads in the data
    df = pd.read_csv("pipeline/data/processed_data/new_mimic.csv")

    #Calls the generate smote_gan function from the smote_gan module
    smote_gan(df)

    #Calls the generate autoencoder function from the autoencoder module
    generate_autoencoder_data(df)

    #Calls the generate_flow_model function from the Flow_Models module
    generate_flow_model(df)

    #Calls the generate_mcmc function from the MCMc module
    generate_mcmc(df)

    #Calls the generate_var_encoders function from the VarEncoders module
    generate_var_encoders(df)

    #Calls the generate_comparative_gen function from the comparative_gen module
    generate_comparative_gen(df)

    #Calls the generate_ground_truth function from the ground_truth module
    generate_ground_truth(df)

    #Calls the generate_lstm function from the LSTM module
    generate_lstm(df) #NOT DONE YET
