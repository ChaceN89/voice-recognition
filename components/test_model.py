from dash import html, dcc
import globals
import dash_bootstrap_components as dbc

layout = html.Div(
    className="",
    children=[
        html.H2("Test Model"),
        html.Div(
            className="side-by-side",
            children=[
                dcc.Dropdown(
                    id='select-model',
                    options=[
                        {'label': 'Option 1', 'value': 'OPT1'},
                        {'label': 'Option 2', 'value': 'OPT2'},
                        {'label': 'Option 3', 'value': 'OPT3'}
                    ],
                    value='OPT1',  # Default value
                    clearable=False,  # Prevents the user from clearing the selection
                    className="dcc-dropdown"  # Add this class for styling
                ),
                dbc.Button("Test", id="test-button", className="button select-button"),
            ]
        ),
        html.Div(
            children=[
                html.Label("Access Granted or not: "),
                html.I(className="fas fa-check"),
                html.Div("/"),
                html.I(className="fas fa-x")
            ]
        )

    ]
)