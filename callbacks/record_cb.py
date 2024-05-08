import io
from dash import html, Input, Output, State, callback_context
import numpy as np
import soundfile as sf
import base64

# Define the audio_samples list at the global level
audio_samples = []

def register(app):
    # to start the recording 
    @app.callback(
        Output("audio-recorder", "recording"),
        Input("record-button", "n_clicks"),
        Input("stop-button", "n_clicks"),
        State("audio-recorder", "recording"),
        prevent_initial_call=True
    )
    def control_recording(record_clicks, stop_clicks, recording):
        if record_clicks is None:
            record_clicks = 0
        if stop_clicks is None:
            stop_clicks = 0
        return record_clicks > stop_clicks

    # handles the recording indicator
    @app.callback(
        Output("recording-indicator", "children"),
        Input("record-button", "n_clicks"),
        Input("stop-button", "n_clicks"),
        State("audio-recorder", "recording"),
        prevent_initial_call=True
    )
    def update_recording_indicator(record_clicks, stop_clicks, recording):
        if record_clicks is None:
            record_clicks = 0
        if stop_clicks is None:
            stop_clicks = 0
        
        # If recording has started, show the recording icon, else hide it
        if record_clicks > stop_clicks:
            return html.Div(className="record-loader")
        else:
            return ""

    # Handles playing the audio for the user using the audio samples
    @app.callback(
        Output("audio-output", "children"),
        Input("play-button", "n_clicks"),
        prevent_initial_call=True
    )
    def play_audio(play_clicks):
        global audio_samples
        if play_clicks:
            if audio_samples:
                # Convert the recorded audio samples to a playable format
                audio_array = np.array(audio_samples)
                with io.BytesIO() as wav_buffer:
                    sf.write(wav_buffer, audio_array, 16000, format="WAV")
                    wav_bytes = wav_buffer.getvalue()
                    wav_base64 = base64.b64encode(wav_bytes).decode()
                    audio_src = f"data:audio/wav;base64,{wav_base64}"
                    return html.Audio(src=audio_src, controls=True)
        return ""


    @app.callback(
        Output("dummy-output", "children"),
        Output("audio-output", "children", allow_duplicate=True),
        Input("audio-recorder", "audio"),
        Input("reset-button", "n_clicks"),
        prevent_initial_call=True, 
    )
    def handle_audio_callbacks(audio, reset_clicks):
        global audio_samples
        ctx = callback_context
        if not ctx.triggered:
            return "",""

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == "audio-recorder" and audio is not None:
            # Update the audio samples with the new audio data
            audio_samples += list(audio.values())
        elif trigger_id == "reset-button" and reset_clicks:
            # Clear the audio samples list to reset the audio
            audio_samples = []
        return "",""