# %%
import threading
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
import socketio

# %%
# Set up a client socket
sio = socketio.Client()
data = []

# DataFrame to display the data
df = pd.DataFrame(data, columns=['Loss', 'Accuracy'])
table_display = widgets.Output()

# Update the DataFrame and refresh the table display
def update_table(loss, accuracy):
    global df
    new_row = pd.DataFrame({'Loss': [loss], 'Accuracy': [accuracy]})
    df = pd.concat([df, new_row], ignore_index=True)
    with table_display:
        display(df.tail())  # Display last few rows of the DataFrame


# Define the event handler for new data
@sio.event
def connect():
    print("Connected to the server.")

@sio.event
def disconnect():
    print("Disconnected from server.")

@sio.event
def training_update(data):
    print("update")
    update_table(data['loss'], data['accuracy'])

# Start the WebSocket connection in a background thread
def start_socket():
    sio.connect('http://127.0.0.1:5000')  # Replace with your server address

thread = threading.Thread(target=start_socket)
thread.start()

# Display the table
display(table_display)

# %%

from IPython.display import HTML

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>HTML Display in Jupyter Notebook</title>
</head>
<body>
    <h1>Hello, Jupyter Notebook!</h1>
    <p>This is an example of displaying HTML in a Jupyter Notebook cell.</p>
</body>
</html>
"""

display(HTML(html_content))
