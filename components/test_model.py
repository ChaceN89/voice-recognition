from dash import html, dcc
import dash_bootstrap_components as dbc

layout = html.Div(
    className="",
    children=[
        html.H2("Test Model"),
        html.Div(
            className="side-by-side",
            children=[
                dcc.Dropdown(
                    id='select-model-dropdown',
                    options=[],  # Options will be populated dynamically
                    value=None,  # No default value initially
                    clearable=False,  # Prevents the user from clearing the selection
                    className="dcc-dropdown"  # Add this class for styling
                ),
                dbc.Button("Test", id="test-button", className="button select-button"),
            ]
        ),
        html.Div(id="test-output")
    ]
)