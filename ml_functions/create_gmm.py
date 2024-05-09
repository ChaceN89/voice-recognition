import os
import librosa
from sklearn.mixture import GaussianMixture
import globals
import pickle
import base64
import io
import os
import numpy as np
import soundfile as sf
import time

# -------------------------------------------------------------------------------
# ----------------- create gmm model and get features ---------------------------
# -------------------------------------------------------------------------------

def create_gmm_model(audio_array, sample_rate):
    features = extract_mfcc_features(audio_array, sample_rate)
    gmm = train_gmm(features)
    return gmm

# Function to extract MFCC features and return the array of features
def extract_mfcc_features(audio_array, sample_rate):
    return [librosa.feature.mfcc(y=audio_array, sr=sample_rate)] # need to return an array t make it work in the training

# This function trains a gmm model using a default 20 Gaussian components using training data and returns the Model
# it assume the training data is a list of features
def train_gmm(training_data, NUMGCOMPONENTS=20):
    # Create the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=NUMGCOMPONENTS)
    
    # Perform the training using the training data samples
    for sample in training_data:
        gmm.fit(sample)

    return gmm

# -------------------------------------------------------------------------------
# --------------------------- Extract Audio Data --------------------------------
# -------------------------------------------------------------------------------

# extract audio array and sample rate (default sample rate)
def extract_base64(src):
    audio_data = src.split(",")[1]
    # Decode base64 data
    audio_bytes = base64.b64decode(audio_data)
    # Create a file-like object for reading binary data
    audio_io = io.BytesIO(audio_bytes)
    # Read audio data using soundfile
    audio_array, sample_rate = sf.read(audio_io)
    return audio_array, sample_rate 


# -------------------------------------------------------------------------------
# ------------------------- Serialize the GMM model -----------------------------
# -------------------------------------------------------------------------------

# Serialize the GMM model and save it to a file
def save_model(model, model_name):
    # Create the directory if it doesn't exist
    if not os.path.exists(globals.audio_model_folder):
        os.makedirs(globals.audio_model_folder)
    
    model_path = os.path.join(globals.audio_model_folder, model_name)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

# get the model from the gloab audio model folder
def fetch_model(model_name):
    model_path = os.path.join(globals.audio_model_folder, model_name)
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        return None
    

# -------------------------------------------------------------------------------
# --------------------------- Using the GMM model -------------------------------
# -------------------------------------------------------------------------------


def test_against_model(gmm_model, audio_array, sample_rate):
    features = extract_mfcc_features(audio_array, sample_rate)
    print(features)
    print(gmm_model)
    scores = test_data_against_gmm(gmm_model, features )

    print("Scores")
    print(scores)
    return "got to scores"



# this takes the socres and flattens then to be returned to calling function
def test_data_against_gmm(gmm_model, test_data_freatures):
  

    # test the sample data
    test_scores = test_samples(gmm_model, test_data_freatures)

    # Flatten score and prediction lists
    flat_test_scores = [i for ix in test_scores for i in ix]

    return flat_test_scores


# this function take a model and test samples and returns there scores and labels
# the labels are not used as a different methods to find the predictions and labels was used
def test_samples(gmm_model, test_samples):
    # Make array lists for predictions and scores
    test_scores = []

    # Iterate through test samples:
    for test_sample in test_samples:

        # Collect prediction scores
        test_score = gmm_model.score_samples(test_sample)

        # Append to result lists
        test_scores.append(test_score)

    return test_scores