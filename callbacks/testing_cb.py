import os
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

import globals
from dash.exceptions import PreventUpdate



from ml_functions import create_gmm

def register(app):
    
    @app.callback(
        Output('select-model-dropdown', 'options'),
        [Input('dropdown-refresh', 'n_intervals')]
    )
    def update_dropdown_options(n_intervals):
        # List all files in the specified folder
        files = os.listdir(globals.audio_model_folder)

        # Generate options for the dropdown
        options = [{'label': file, 'value': file} for file in files]
        
        return options
    

    @app.callback(
        Output('test-output', 'children'),
        [Input('test-button', 'n_clicks'),],
        [
            State('select-model-dropdown', 'value'),
            State("audio-output", "children")
        ]
    )
    def test_model(n_clicks, selected_model, audio_output):
        if n_clicks:
            if not audio_output:
                return "Need to record testing audio."
            if not selected_model:
                return "Need to select a model."
    
            # get the model
            gmm_model = create_gmm.fetch_model(selected_model)
            if not gmm_model:
                return "Can't find model."

            src = audio_output['props'].get('src', '')
            if src.startswith("data:audio/wav;base64,"):
                # extract base 64 array and sample rate for the testing data
                audio_array, sample_rate = create_gmm.extract_base64(src)
            
                create_gmm.test_against_model(gmm_model, audio_array, sample_rate)
                
                



            # change return to show access granted or not
            return html.Div(
                children=[
                    html.Label("Access Granted or not: "),
                    html.I(className="fas fa-check"),
                    html.Div("////"),
                    html.I(className="fas fa-x")
                ]
            )
        else:
            raise PreventUpdate 
    

    


