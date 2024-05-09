from dash import html, dcc
import globals
import dash_bootstrap_components as dbc

# https://pypi.org/project/dash-recording-components/

layout = html.Div(
    children=[
        html.H2("Create Model"),
        html.Div(
            className="side-by-side",
            children=[
                dcc.Input(id='model-name-input', type='text', placeholder='Name Model...'),
                html.Button("Create Model", id="create-model"),
            ]
        ),
        html.Div( id="output-container" ),

        html.H2("Delete Model"),
        html.Div(
            className="side-by-side",
            children=[
                dcc.Dropdown(
                    id='delete-model-dropdown',
                    options=[
                        {'label': 'Option 1', 'value': 'OPT1'},
                        {'label': 'Option 2', 'value': 'OPT2'},
                        {'label': 'Option 3', 'value': 'OPT3'}
                    ],
                    value='OPT1',  # Default value
                    clearable=False,  # Prevents the user from clearing the selection
                    className="dcc-dropdown"  # Add this class for styling
                ),
                dbc.Button("Delete", id="delete-button", className="button delete-button"),
            ]
        )


    ],
)