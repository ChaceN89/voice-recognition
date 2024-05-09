from dash import Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
from ml_functions import create_gmm

def register(app):
    @app.callback(
        Output("output-container", "children"),  # Update some output container
        Input("create-model", "n_clicks"),      # Button click event
        State("model-name-input", "value"),     # Model name input
        State("audio-output", "children"),      # Audio output
        prevent_initial_call=True
    )
    def process_model_creation(n_clicks, model_name, audio_output):
        if n_clicks:
            if not audio_output:
                return "Need to record training audio (Press play button before creating profile)."
            if not model_name:
                return "Need to provide profile name."

            # get the audio infomation out of the dict 
            src = audio_output['props'].get('src', '')
            if src.startswith("data:audio/wav;base64,"):

                # extract base 64 array and sample rate
                audio_array, sample_rate = create_gmm.extract_base64(src)
                gmm_model, features = create_gmm.create_gmm_model(audio_array, sample_rate) 

                # save model in the file structure under audio_models folder
                create_gmm.save_model(gmm_model, features, model_name)
                
                return f"Profile '{model_name}' created successfully."
            else:
                return "Failed to create profile. Could not read audio."

        else:
            raise PreventUpdate  # Callback is not triggered until button is clicked


