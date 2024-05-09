import os
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture

import globals
import pickle


def create_gmm_model(audio_array, sample_rate):
    features = extract_mfcc_features(audio_array, sample_rate)
    gmm = train_gmm(features)
    return gmm

# Function to extract MFCC features and return the array of features
def extract_mfcc_features(audio_array, sample_rate):
    return [librosa.feature.mfcc(y=audio_array, sr=sample_rate)] # need to return an array t make it work


# This function trains a gmm model using a default 20 Gaussian components using training data and returns the Model
# it assume the training data is a list of features
def train_gmm(training_data, NUMGCOMPONENTS=20):
    # Create the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=NUMGCOMPONENTS)
    
    # Perform the training using the training data samples
    for sample in training_data:
        gmm.fit(sample)

    return gmm



# Serialize the GMM model and save it to a file
def save_model(model, model_name):
    # Create the directory if it doesn't exist
    if not os.path.exists(globals.audio_model_folder):
        os.makedirs(globals.audio_model_folder)
    
    model_path = os.path.join(globals.audio_model_folder, model_name)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


