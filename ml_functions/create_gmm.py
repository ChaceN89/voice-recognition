import os
import librosa
from scipy import stats
from sklearn.mixture import GaussianMixture
import globals
import pickle
import base64
import io
import os
import numpy as np
import soundfile as sf
from scipy.stats import norm
import plotly.graph_objs as go
import numpy as np
from dash import html

# -------------------------------------------------------------------------------
# ----------------- create gmm model and get features ---------------------------
# -------------------------------------------------------------------------------

def create_gmm_model(audio_array, sample_rate):
    features = extract_mfcc_features(audio_array, sample_rate)
    gmm = train_gmm(features)
    return gmm, features

# Function to extract MFCC features and return the array of features
def extract_mfcc_features(audio_array, sample_rate):
    mfcc_features = librosa.feature.mfcc(y=audio_array, sr=sample_rate)
    mfcc_features = mfcc_features.T  # Transpose to match the expected format
    print("MFCC Features shape:", mfcc_features.shape)
    return [mfcc_features]

# This function trains a gmm model using a default 20 Gaussian components using training data and returns the Model
# it assume the training data is a list of features
def train_gmm(training_data, NUMGCOMPONENTS=20):
    # Create the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=NUMGCOMPONENTS)
    
    # Perform the training using the training data samples
    for sample in training_data:
        print("Training sample shape:", sample.shape)
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
def save_model(model, features, model_name):
    # Create the directory if it doesn't exist
    if not os.path.exists(globals.audio_model_folder):
        os.makedirs(globals.audio_model_folder)
    
    model_path = os.path.join(globals.audio_model_folder, model_name)
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'features': features}, f)

# Get the model from the global audio model folder
def fetch_model(model_name):
    model_path = os.path.join(globals.audio_model_folder, model_name)
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['features']
    else:
        return None, None
    

# -------------------------------------------------------------------------------
# --------------------------- Using the GMM model -------------------------------
# -------------------------------------------------------------------------------


def test_against_model(gmm_model, auth_features, audio_array, sample_rate):
    features = extract_mfcc_features(audio_array, sample_rate)
    print("Test features shape:", features[0].shape)
    auth_scores = test_data_against_gmm(gmm_model, auth_features)
    scores = test_data_against_gmm(gmm_model, features)
    print("Scores:", scores)

    x = np.arange(-200, 200, 1)

    auth_Mu = np.mean(auth_scores)
    auth_Std = np.std(auth_scores)
    auth_Prob = norm.pdf(x, loc=auth_Mu, scale=auth_Std)

    test_mu = np.mean(scores)
    test_std = np.std(scores)
    test_Prob = norm.pdf(x, loc=test_mu, scale=test_std)

    print("auth_Mu ", auth_Mu)
    print("auth_std ", auth_Std)
    print("test_mu ", test_mu)
    print("test_std ", test_std)
  
    # test for a match 
    is_accepted, mean_diff, std_diff = test_with_thresholds(test_mu, test_std, auth_Mu, auth_Std, globals.mean_threshold, globals.std_threshold, globals.CI)


    # get the plotting infomation
    plot = create_plot(auth_Prob, auth_scores, auth_Mu, auth_Std,
                        test_Prob, scores, test_mu, test_std )

    # set up return infomation 
    message = ""
    if is_accepted:
        message = f"Audio Accepted at a confidence interval of: {globals.CI*100}%"
    else:
        message = f"Audio Rejected at a confidence interval of: {globals.CI*100}%."
    
    icon_class = "fas fa-check" if is_accepted else "fas fa-x"

    
    # change return to show access granted or not based on output of test agaisnt model
    return html.Div(
            className="access-text",
            children=[
                html.I(className=icon_class, style={'fontSize': '34px'}),
                html.Div(
                    children=[
                        html.Label(message),
                        html.Ul([
                            html.Li(f"Profile Mean: {round(auth_Mu, 2)}"),
                            html.Li(f"Profile STD: {round(auth_Std, 2)}"),
                            html.Li(f"Test Mean: {round(test_mu, 2)}"),
                            html.Li(f"Test STD: {round(test_std, 2)}"),
                            html.Li(f"Mean Difference: {round(mean_diff, 2)}"),
                            html.Li(f"STD Difference: {round(std_diff, 2)}"),
                        ])
                    ]
                )
            ]
        ), plot


# This takes the scores and flattens them to be returned to calling function
def test_data_against_gmm(gmm_model, test_data_features):
    # Test the sample data
    test_scores = test_samples(gmm_model, test_data_features)
    # Flatten score and prediction lists
    flat_test_scores = [i for ix in test_scores for i in ix]
    return flat_test_scores

# This function takes a model and test samples and returns their scores and labels
def test_samples(gmm_model, test_samples):
    # Make array lists for predictions and scores
    test_scores = []
    # Iterate through test samples:
    for test_sample in test_samples:
        print("Test sample shape:", test_sample.shape)
        # Collect prediction scores
        test_score = gmm_model.score_samples(test_sample)
        # Append to result lists
        test_scores.append(test_score)
    return test_scores


# -------------------------------------------------------------------------------
# --------------------------- testing with thresholds -----------------------------
# -------------------------------------------------------------------------------

# Function to calculate z-score
def calculate_z_score(test_mu, real_mu, real_std):
    return (test_mu - real_mu) / real_std

# Function to normalize the scores
def normalize_scores(scores, mean, std):
    return (scores - mean) / std

# Function to test with thresholds on mean and standard deviation differences
def test_with_thresholds(test_mu, test_std, real_mu, real_std, mean_threshold, std_threshold, CI):
    mean_diff = abs(test_mu - real_mu)
    std_diff = abs(test_std - real_std)

    print("mean_diff ", mean_diff)
    print("std_diff ", std_diff)
    print("mean_threshold ", mean_threshold)
    print("std_threshold ", std_threshold)

    # see if it can get past the thresholds    # 
    if mean_diff > mean_threshold or std_diff > std_threshold:
        print("reject 1")
        return False, mean_diff, std_diff
    
    # Normalize the scores
    norm_test_mu = normalize_scores(test_mu, real_mu, real_std)
    norm_test_std = normalize_scores(test_std, real_mu, real_std)
    norm_real_mu = normalize_scores(real_mu, real_mu, real_std)
    norm_real_std = normalize_scores(real_std, real_mu, real_std)
    
    # Calculate the z-score
    z_score = calculate_z_score(norm_test_mu, norm_real_mu, norm_real_std)
    
    # z_score = calculate_z_score(test_mu, real_mu, real_std)
    alpha = 1 - CI
    critical_value = stats.norm.ppf(1 - alpha / 2)
    
    if abs(z_score) <= critical_value:
        print("accept 1")
        return True, mean_diff, std_diff
    else:
        print("reject 2")
        return False, mean_diff, std_diff


# -------------------------------------------------------------------------------
# ----------------------------- creating plots  ---------------------------------
# -------------------------------------------------------------------------------


def create_plot(auth_Prob, auth_scores, auth_Mu, auth_Std, test_Prob, test_scores, test_mu, test_std):
    """
    Create a plot with bell curves for both the authentic and testing data.

    Parameters:
    auth_Prob (array-like): Probability values for the authentic data.
    auth_scores (array-like): Score values for the authentic data.
    auth_Mu (float): Mean score of the authentic data.
    auth_Std (float): Standard deviation of the authentic data.
    test_Prob (array-like): Probability values for the test data.
    test_scores (array-like): Score values for the test data.
    test_mu (float): Mean score of the test data.
    test_std (float): Standard deviation of the test data.

    Returns:
    fig (plotly.graph_objs.Figure): Plotly figure object with the bell curves.
    """

    # Generate x values for the bell curves
    x_values = np.linspace(min(min(auth_scores), min(test_scores)), max(max(auth_scores), max(test_scores)), 500)

    # Calculate the bell curves using the normal distribution
    auth_bell_curve = (1 / (auth_Std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - auth_Mu) / auth_Std) ** 2)
    test_bell_curve = (1 / (test_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - test_mu) / test_std) ** 2)

    # Create the figure
    fig = go.Figure()

    # Add the bell curve for authentic data
    fig.add_trace(go.Scatter(
        x=x_values,
        y=auth_bell_curve,
        mode='lines',
        name='Authentic Data',
        line=dict(color='blue')
    ))

    # Add the bell curve for test data
    fig.add_trace(go.Scatter(
        x=x_values,
        y=test_bell_curve,
        mode='lines',
        name='Test Data',
        line=dict(color='red')
    ))

    # Add the actual data points as scatter plots for additional context
    fig.add_trace(go.Scatter(
        x=auth_scores,
        y=auth_Prob,
        mode='markers',
        name='Authentic Scores',
        marker=dict(color='blue', symbol='circle')
    ))

    fig.add_trace(go.Scatter(
        x=test_scores,
        y=test_Prob,
        mode='markers',
        name='Test Scores',
        marker=dict(color='red', symbol='circle')
    ))

    # Update the layout of the plot
    fig.update_layout(
        title='Authentic vs Test Data Bell Curves',
        xaxis_title='Scores',
        yaxis_title='Probability Density',
        legend_title='Data Type',
        template='plotly_white'
    )

    return fig