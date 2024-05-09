from dash import html, dcc
import dash_recording_components as drc

layout = html.Div(
    className="set-audio",
    children=[
        html.H1("Audio Recorder and Player"),

        html.P("Here you can record audio to create a vocal profile or test against an existing vocal profile."),
        html.Div(
            className="help-text",
            children=[
                html.Div(
                    children=[
                        html.H2("Instructions:"),
                        html.Ul([
                            html.Li("Creating a Profile: Record for about 1 minute to capture sufficient audio for creating a profile."),
                            html.Li("Testing Audio: Record for about 10 seconds to test the audio against an existing profile."),
                            html.Li("Controls:"),
                            html.Ul([
                                html.Li("Pause/Resume: You may pause and resume recording as needed."),
                                html.Li("Play: Use the play button to review your recording."),
                                html.Li("Reset: Use the reset button to start over if necessary.")
                            ])
                        ]),
                    ]
                ),
                html.Div(
                    children=[
                        html.H2("Profile Management:"),
                        html.Ul([
                            html.Li("Create Vocal Profile: Once the audio is ready, enter a profile name and click 'Create Profile' to create the vocal profile."),
                            html.Li("Delete Vocal Profile: You can delete a profile at any time by selecting it from the dropdown and clicking 'Delete'."),
                            html.Li("Test Vocal Profile: Record new audio and test it against your created vocal profile by selecting the profile and clicking 'Test'.")
                        ])
                    ]
                )
            ]
        ),
        
        



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


