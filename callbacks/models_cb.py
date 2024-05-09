import io
import os
from dash import html, Input, Output, State, callback_context
import numpy as np
import soundfile as sf
import base64
import time
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
            # Check if model name and audio output are provided
            if model_name and audio_output:
                # get the audio infomation out of the dict 
                src = audio_output['props'].get('src', '')
                if src.startswith("data:audio/wav;base64,"):
                    # Extract base64-encoded audio data
                    audio_data = src.split(",")[1]
                    # Decode base64 data
                    audio_bytes = base64.b64decode(audio_data)
                    # Create a file-like object for reading binary data
                    audio_io = io.BytesIO(audio_bytes)
                    # Read audio data using soundfile
                    audio_array, sample_rate = sf.read(audio_io)
                    
                    gmm_model = create_gmm.create_gmm_model(audio_array, sample_rate) 
                    print(gmm_model)

                    # save model in the file structure under audio_models folder
                    create_gmm.save_model(gmm_model, model_name)
                    
                    return f"Model '{model_name}' created successfully."
                else:
                    return "Failed to create model. Could not read audio."
            else:
                return "Please provide both model name and audio (Press play before creating model)."
        else:
            raise PreventUpdate  # Callback is not triggered until button is clicked


