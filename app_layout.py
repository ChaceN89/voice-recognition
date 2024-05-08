# app_layout.py
# the main layout html for the app
# call multipe layout components

from dash import html, dcc
from components import create_model, test_model

layout = html.Div(
    className="container",
    children=[

        # the header with report settings
        html.Header(
             html.H1("Header")
        ),

        # the plots and tables
        html.Div(
            className="body",
            children=[
                create_model.layout,
                test_model.layout
            ]
        ),

       html.Footer(
            html.H3("Footer")
       )
    ]
)