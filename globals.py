
# to set a auto start when running main.py
host = 'http://127.0.0.1:8050/'

audio_model_folder = "audio_models"

CI=0.95
# a 95% confidence level is less likely to accept the test case compared to a 90% confidence level. Here's why:
# A 95% confidence level corresponds to a larger critical value (approximately 1.96 for a two-tailed test).
# A 90% confidence level corresponds to a smaller critical value (approximately 1.645 for a two-tailed test).

# values to initially set a threshold value foe max difference in mean and std allowed before rejecting
mean_threshold = 40  
std_threshold = 2

# the CI and z score test is normalized so it will be accepted more often 