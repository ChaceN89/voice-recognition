# main.py
# the driver file for the application
# using styles in assets folder is the default behaviour ofn dash apps

# import libraires
import dash
from threading import Timer
import webbrowser
import os

import app_layout 
import globals
from callbacks import record_cb

# Function to automatically open the web browser when the app is launched
def open_browser():
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new(globals.host)

# Create the Dash app
app = dash.Dash(__name__)

# Add Font Awesome for icons
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

# use layout file to set up application layout
app.layout = app_layout.layout

record_cb.register(app)

# Run the app (will automatically open a window)
if __name__ == '__main__':
    Timer(1, open_browser).start()  # Automatically open the application
    app.run_server(debug=True, use_reloader=True)
