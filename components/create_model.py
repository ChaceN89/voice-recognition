from dash import html, dcc
import globals
import dash_bootstrap_components as dbc


layout = html.Div(
    children=[
        html.H2("Create Model"),
        dbc.Button(html.I(className="fas fa-microphone-slash"), id="record-button-create", className="mic-button"),


        html.H3("Delete Model"),
        dcc.Dropdown(
            id='delete-model-dropdown',
            options=[
                {'label': 'Option 1', 'value': 'OPT1'},
                {'label': 'Option 2', 'value': 'OPT2'},
                {'label': 'Option 3', 'value': 'OPT3'}
            ],
            value='OPT1',  # Default value
            clearable=False  # Prevents the user from clearing the selection
        ),
        dbc.Button("Delete", id="delete-button", className="delete-button"),


    ],
)