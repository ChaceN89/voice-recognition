import librosa
import numpy as np
from sklearn.mixture import GaussianMixture




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