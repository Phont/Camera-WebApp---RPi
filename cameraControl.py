#Framework of Motion Detection code from pyimagesearch, expanded by Andrew Amerman
from pyimagesearch.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import imutils
import cv2
import time

outputFrame = None
cameraAction = 0
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
	# default page
	global cameraAction 
	cameraAction = 0
	return render_template("indexBase.html")

def detect_motion(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock, cameraAction

	# initialize the motion detector and the total number of frames
	# read thus far
	md = SingleMotionDetector(accumWeight=0.1)
	total = 0

	# loop over frames from the video stream
	while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7, 7), 0)

		# if the total number of frames has reached a sufficient
		# number to construct a reasonable background model, then
		# continue to process the frame
		if total > frameCount:
			# detect motion in the image
			motion = md.detect(gray)
			# check to see if motion was found in the frame
			if motion is not None:
				# unpack the tuple and draw the box surrounding the
				# "motion area" on the output frame
				(thresh, (minX, minY, maxX, maxY)) = motion
				if cameraAction == 5:
					cv2.rectangle(frame, (minX, minY), (maxX, maxY),
					(	0, 0, 255), 2)
		
		# update the background model and increment the total number
		# of frames read thus far
		md.update(gray)
		total += 1
		
		#if cameraAction == 0: do nothing
		
		if cameraAction == 1: #rotate left
			frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
		
		elif cameraAction == 2: #rotate right
			frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
		
		elif cameraAction == 3: #flip
			frame = cv2.rotate(frame, cv2.ROTATE_180)
		
		elif cameraAction == 4: #convert to greyscale
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
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
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")
		
@app.route('/rotate-left')
def rotate_left():
	global cameraAction 
	cameraAction = 1
	
	return render_template('rotate-left.html')
	
@app.route('/rotate-right')
def rotate_right():
	global cameraAction 
	cameraAction = 2
	
	return render_template("rotate-right.html")
	
@app.route('/flip-image')
def flip_image():
	global cameraAction 
	cameraAction = 3
	
	return render_template("flip-image.html")
	
@app.route('/grey-scale/')
def grey_scale():
	global cameraAction 
	cameraAction = 4
	
	return render_template("grey-scale.html")
	
@app.route('/motion-detection/')
def motion_detection():
	global cameraAction 
	cameraAction = 5
	
	return render_template("motion-detection.html")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)
	

# release the video stream pointer
vs.stop()
