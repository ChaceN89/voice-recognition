import os
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

import globals


def register(app):
    
    @app.callback(
        Output('delete-model-dropdown', 'options'),
        [Input('dropdown-refresh', 'n_intervals')]
    )
    def update_dropdown_options(n_intervals):
        # List all files in the specified folder
        files = os.listdir(globals.audio_model_folder)

        # Generate options for the dropdown
        options = [{'label': file, 'value': file} for file in files]
        
        return options
    
    @app.callback(
        Output('delete-model-text', 'children'),
        [Input('delete-button', 'n_clicks')],
        [State('delete-model-dropdown', 'value')]
    )
    def delete_file(n_clicks, selected_value):
        print(n_clicks)
        print(selected_value)
        if n_clicks:
            if selected_value is None:
                return "Select a Model"

            # Delete the selected file
            file_path = os.path.join(globals.audio_model_folder, selected_value)
            os.remove(file_path)

            return f"{selected_value} Model Deleted"
        else:
            return f""
