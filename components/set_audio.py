from dash import html
import dash_recording_components as drc


layout = html.Div(
    className="set-audio",
    children=[
        html.H1("Audio Recorder and Player"),
        html.P("For creating a model, record for about 1 minute. For testing audio against a model, record for about 10 seconds."),
        html.Div(
            className="side-by-side tight",
            children=[
                html.Button(
                    className="button",
                    id="record-button",
                    children=[ html.I(className="fas fa-microphone"), "Record" ]
                ),
                html.Button(
                    className="button",
                    id="stop-button", 
                    n_clicks=0,
                    children=[ html.I(className="fas fa-microphone-slash"), "Stop Recording" ]
                ),
                html.Button(
                    className="button",
                    id="play-button",
                    children=[ html.I(className="fas fa-play"), "Play" ]
                ),
                html.Button(
                    className="button",
                    id="reset-button",
                    children=[ html.I(className="fas fa-undo"), "Rest" ]
                ),

            ]
        ),

        html.Div(id="audio-output"),
        html.Div(id="dummy-output", style={"display": "none"}),
        html.Div( id="recording-indicator" ),
        drc.AudioRecorder(id="audio-recorder")
    ]
)


