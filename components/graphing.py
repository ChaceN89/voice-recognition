from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
# Create a DataFrame with dummy data
df = pd.DataFrame({
    "x": [1, 2, 3, 4, 5],
    "y": [10, 11, 12, 13, 14]
})

fig = px.line(df, x="x", y="y", title="Dummy Line Plot")


layout = html.Div(
    className="graphing",
    children=[
        html.H1("Dummy Graph Example"),
        html.Div(
            dcc.Graph(id='dummy-graph', figure=fig, className='dummy-graph'),
            className='graph-container'
        ),
        html.Hr(),  # Horizontal line for separation
        html.P("This is a dummy graph showing a simple line plot.")
    ]
)