from dash import html
import dash_recording_components as drc


layout = html.Div(
    className="set-audio",
    children=[
        html.H1("Audio Recorder and Player"),
        html.Button("Record", id="record-button"),
        html.Button("Stop Recording", id="stop-button", n_clicks=0),
        html.Button("Play", id="play-button"),
        html.Button("Rest", id="reset-button"),

        html.Div(id="audio-output"),
        html.Div(id="dummy-output", style={"display": "none"}),
        drc.AudioRecorder(id="audio-recorder")
    ]
)


