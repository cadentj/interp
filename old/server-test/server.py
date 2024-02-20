# This part remains the same as the previous example
from flask import Flask
from flask_socketio import SocketIO, emit
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

def background_thread():
    """Simulates sending data to client."""
    count = 0
    while True:
        time.sleep(3)
        # Emit the training_update event
        socketio.emit('training_update', {'one': 1, 'two' :1})

@app.route('/')
def index():
    return "Training Server"

if __name__ == '__main__':
    thread = threading.Thread(target=background_thread)
    thread.daemon = True
    thread.start()
    socketio.run(app, debug=True)
