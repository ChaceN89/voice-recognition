# app_layout.py
# the main layout html for the app 
# Header, body template and footer
# body contains logical componets 

from dash import html
from components import set_models, test_model, set_audio, graphing, delete_models

layout = html.Div(
    className="container",
    children=[
        html.Header( # the header with report settings
             html.H1("Gaussian Mixture Model Voice Recognition")
        ),
        html.Div(
            className="body-container",
            children=[
                html.Template( # the plots and tables
                    children=[
                        set_audio.layout,
                        set_models.layout,
                        delete_models.layout,
                        test_model.layout,
                        graphing.layout
                    ]
                ),
            ]
        ),
        html.Footer( # Footer for future reference 
            children=[
                html.H3("Chace Nielson"),
                html.A('Portfolio', href='https://chacen89.github.io/Portfolio-ChaceNielson/', target='_blank')
            ]
        )
    ]
)