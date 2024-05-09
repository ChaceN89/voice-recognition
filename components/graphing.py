from dash import html, dcc

layout = html.Div(
    className="graphing",
    children=[
        html.Hr(),
        html.Div(
            dcc.Graph(id='outcome-graph', className='dummy-graph'),
            className='graph-container'
        ),
        html.Hr(),  # Horizontal line for separation
        html.P("This graph shows the results of the model data(authentic) and the testing data.")
    ]
)