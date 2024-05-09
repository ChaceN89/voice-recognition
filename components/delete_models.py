from dash import html, dcc
import dash_bootstrap_components as dbc

layout = html.Div(
    children=[
        html.H2("Delete Model"),
        html.Div(
            className="side-by-side",
            children=[
                dcc.Dropdown(
                    id='delete-model-dropdown',
                    options=[],  # Options will be populated dynamically
                    value=None,  # No default value initially
                    clearable=False,  # Prevents the user from clearing the selection
                    className="dcc-dropdown"  # Add this class for styling
                ),
                dbc.Button("Delete", id="delete-button", className="button delete-button"),
            ]
        ),
        html.Div(id="delete-model-text"),
    ],
)