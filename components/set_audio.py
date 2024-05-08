from dash import html, dcc
import dash_recording_components as drc

layout = html.Div(
    className="set-audio",
    children=[
        html.H1("Audio Recorder and Player"),
        html.P("For creating a model, record for about 1 minute. For testing audio against a model, record for about 10 seconds. You may pause recording and then resume recording."),
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
        
        html.Div(id="audio-output"), # stores the output of the audio, contains value(html.audio) or is null
        html.Div(id="dummy-output", style={"display": "none"}),
        html.Div( id="recording-indicator" ),
        html.Div( id="recording-timer" ),
        html.Div( id="recording-stored" ),
        drc.AudioRecorder(id="audio-recorder"),
        dcc.Interval(id='interval-component', interval=1000, n_intervals=0),  # Update every 1 second

    ]
)


