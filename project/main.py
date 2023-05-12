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
from preprocess_data import preprocess_data, sample_data
from gen_models.smote_gan import smote_gan, plot_original_vs_synthetic_gan
from gen_models.MCMc import mcmc, plot_original_vs_synthetic_mcmc
#Defines the main function for the gen_models module that takes in the new_mimic.csv file from pipeline/data/proccessed_data
def main():
    """
    Calls the various synthethic generation techniques and plots the results
    """
    #Loads in the data
    df = pd.read_csv("pipeline/data/processed_data/merged.csv")

    # Preprocess the data
    df = preprocess_data(df)

    # Randomly sample the data
    sample_size = 10
    random_state = 42
    df_sampled = sample_data(df, sample_size, random_state=random_state)

    #Calls the generate smote_gan function from the smote_gan module
    smote_df = smote_gan(df,'has_sepsis', 1000)


    plot_original_vs_synthetic_gan(df, smote_df, 'has_sepsis')

    #Calls the mcmc function from the MCMc module
    mcmc_df = mcmc(df,'has_sepsis')

    # Plot original data and synthetic data
    plot_original_vs_synthetic_mcmc(df, mcmc_df, 'has_sepsis')


if __name__ == '__main__': 
    main()
