from dash import Dash, html, dcc, Input, Output

def register(app):
    
    # for creation
    @app.callback(
        Output("record-button-create", "children"),
        Output("record-button-create", "className"),
        Input("record-button-create", "n_clicks"),
        prevent_initial_call=True
    )
    def toggle_recording(n_clicks):
        if n_clicks % 2 == 1:  # Recording on
            return html.I(className="fas fa-microphone"), "mic-button recording"
        else:  # Recording off
            return html.I(className="fas fa-microphone-slash"), "mic-button"
        
    # for the testing
    @app.callback(
        Output("record-button-test", "children"),
        Output("record-button-test", "className"),
        Input("record-button-test", "n_clicks"),
        prevent_initial_call=True
    )
    def toggle_recording(n_clicks):
        if n_clicks % 2 == 1:  # Recording on
            return html.I(className="fas fa-microphone"), "mic-button recording"
        else:  # Recording off
            return html.I(className="fas fa-microphone-slash"), "mic-button"



