# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from FrameManipulation import frameFilter
from pyimagesearch.motion_detection import SingleMotionDetector
import FrameManipulation
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template, session
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect
# from adafruit_servokit import ServoKit
import threading
import argparse
import datetime
import imutils
import time
import cv2
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import telegram

bot = telegram.Bot(token="1574500597:AAH0cU4EFdJzrwtxcgSpyy5MA1ijtoowPR4")

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
filter = "none"
lock = threading.Lock()
async_mode = None
thread = None

move = "stop"
lr = 90
ud = 90
# kit = ServoKit(channels=16)
# kit.servo[0].set_pulse_width_range(600,2610)
# kit.servo[1].set_pulse_width_range(600,2610)

# initialize a flask object
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode)

# initialize the video stream and allow the camera sensor to
# warmup
# vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)


def background_thread():
    """Example of how to send server generated events to clients."""
    count = 0
    while True:
        socketio.sleep(10)
        count += 1
        socketio.emit('my_response',
                      {'data': 'Server generated event', 'count': count})


def move_camera():
    # grab global references to the video stream, output frame, and
    # lock variables
    global move, lr, ud

    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it

        time.sleep(.1)
        if (move == "right"):
            lr += 5
            if lr > 135:
                lr = 135
        # kit.servo[1].angle = lr
        elif (move == "left"):
            lr -= 5
            if lr < 45:
                lr = 45
        # kit.servo[1].angle = lr
        elif move == "up":
            ud += 5
            if ud > 135:
                ud = 135
        # kit.servo[0].angle = ud
        elif move == "down":
            if ud < 45:
                ud = 45
            ud -= 5
        # kit.servo[0].angle = ud


# print("ud: " +str(ud) +" lr: "+str(lr))


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def detect_motion(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock

    # initialize the motion detector and the total number of frames
    # read thus far
    md = SingleMotionDetector(accumWeight=0.1)
    total = 0

    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        frame = vs.read()
        frame2 = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # if the total number of frames has reached a sufficient
        # number to construct a reasonable background model, then
        # continue to process the frame
        if total > frameCount:
            # detect motion in the image
            motion = md.detect(gray)

            # cehck to see if motion was found in the frame
            if motion is not None:
                # unpack the tuple and draw the box surrounding the
                # "motion area" on the output frame
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame2, (minX, minY), (maxX, maxY),
                              (0, 0, 255), 2)
                filename = "/home/william/Pictures/" + timestamp.strftime(
                    "%A %d %B %Y %I:%M:%S%p") + ".png"
                cv2.imwrite(filename, frame)
                bot.send_photo(chat_id="1360995944", photo=open(filename, 'rb'))
        # update the background model and increment the total number
        # of frames read thus far
        md.update(gray)
        total += 1

        # acquire the lock, set the output frame, and release the
        # lock
        with lock:

            outputFrame = frame.copy()


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@socketio.event
def my_ping():
    emit('my_pong')


@socketio.event
def connect():
    global thread
    with lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)
    emit('my_response', {'data': 'Connected', 'count': 0})


@socketio.on('move_up')
def move_up(message):
    global move
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_move',
         {'data': message['data'], 'count': session['receive_count']})
    move = message['data']


@socketio.on('move_down')
def move_down(message):
    global move
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_move',
         {'data': message['data'], 'count': session['receive_count']})
    move = message['data']


@socketio.on('move_left')
def move_left(message):
    global move
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_move',
         {'data': message['data'], 'count': session['receive_count']})
    move = message['data']


@socketio.on('move_right')
def move_right(message):
    global move
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_move',
         {'data': message['data'], 'count': session['receive_count']})
    move = message['data']


@socketio.on('move_stop')
def move_stop(message):
    global move
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_move',
         {'data': message['data'], 'count': session['receive_count']})
    move = message['data']


# check to see if this is the main thread of execution
if __name__ == '__main__':
    updater = Updater("1574500597:AAH0cU4EFdJzrwtxcgSpyy5MA1ijtoowPR4", use_context=True)
    dp = updater.dispatcher
    updater.start_polling()


    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_motion, args=(
        32,))  # args["frame_count"],))
    t.daemon = True
    t.start()

    t2 = threading.Thread(target=move_camera, args=(
    ))  # args["frame_count"],))
    t2.daemon = True
    t2.start()

    # start the flask app
    # app.run(host=args["ip"], port=args["port"], debug=True,
    #	threaded=True, use_reloader=False)

    socketio.run(app, host="127.0.0.1", port="5000")

# release the video stream pointer
vs.stop()
