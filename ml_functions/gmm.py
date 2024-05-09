import librosa
from sklearn.mixture import GaussianMixture




def create_gmm_model(audio_array, sample_rate):
    features = extract_mfcc_features(audio_array, sample_rate)
    gmm = train_gmm(features)
    return gmm



def test_audio_vs_gmm():
    pass

    # auth_test_scores, auth_labels = test_data_against_gmm(gmm,features)




# Function to extract MFCC features and return the array of features
def extract_mfcc_features(audio_array, sample_rate):
    return librosa.feature.mfcc(y=audio_array, sr=sample_rate)


def train_gmm(training_data, NUMGCOMPONENTS=20):
    # Create the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=NUMGCOMPONENTS)
    
    # Perform the training using the training data samples
    for sample in training_data:
        gmm.fit(sample)

    return gmm



# this takes the socres and flattens then to be returned to calling function
def test_data_against_gmm(gmm_model, test_data_freatures):
    # store the scores
    scores = []

    # test the sample data
    test_scores, prediction_labels = test_samples(gmm_model, test_data_freatures)

    # Flatten score and prediction lists
    flat_test_scores = [i for ix in test_scores for i in ix]
    flat_prediction_labels = [i for ix in prediction_labels for i in ix]

    return flat_test_scores, flat_prediction_labels


# this function take a model and test samples and returns there scores and labels
# the labels are not used as a different methods to find the predictions and labels was used
def test_samples(gmm_model, test_samples):
    # Make array lists for predictions and scores
    test_scores = []
    prediction_labels = []

    # Iterate through test samples:
    for test_sample in test_samples:

        # Make predictions on test sample
        prediction_label = gmm_model.predict(test_sample)

        # Collect prediction scores
        test_score = gmm_model.score_samples(test_sample)

        # Append to result lists
        test_scores.append(test_score)
        prediction_labels.append(prediction_label)

    return test_scores, prediction_labels