from dash import html, dcc
import globals
import dash_bootstrap_components as dbc

layout = html.Div(
    children=[
        html.H2("Test Model "),

        html.H3("Select Models"),
        dcc.Dropdown(
            id='select-model-dropdown',
            options=[
                {'label': 'Option 1', 'value': 'OPT1'},
                {'label': 'Option 2', 'value': 'OPT2'},
                {'label': 'Option 3', 'value': 'OPT3'}
            ],
            value='OPT1',  # Default value
            clearable=False  # Prevents the user from clearing the selection
        ),


        dbc.Button(html.I(className="fas fa-microphone-slash"), id="record-button-test", className="mic-button"),
    ],
)