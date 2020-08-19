# python calc_ear.py --shape-predictor shape_predictor_68_face_landmarks.dat --fps {frame_rate}

from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import argparse
import math
import progressbar

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear
	
video_path = "R:\\Users\\Rahul\\Documents\\Programming\\Right Way.mp4"

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-f", "--fps", required=True)
args = vars(ap.parse_args())

# Load and initialize video
print("[INFO] Loading video...")
cap = cv2.VideoCapture(video_path)
success = cap.read() 

# Calculate FPS
video_fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
input_fps = int(args['fps']) 
total_frame_count = math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("[INFO] Video FPS is equal to {}".format(video_fps))

# FPS check
err = "Inputted FPS cannot be greater than: {}".format(video_fps)
if (input_fps > video_fps):
    raise Exception(err)

skip_rate = math.ceil(video_fps / input_fps)
# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Grab the indexes of the facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Initialize progress bar
pbar = progressbar.ProgressBar(maxval=100, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

ears = []
fps_count = 0

print("[INFO] Starting read of video...")
pbar.start()
while success:
    # Read next frame
    success, img = cap.read()
    if not success:
        break

    # Skip frames depending on inputted FPS count
    fps_count += 1
    if not fps_count % skip_rate is 0:
        continue

    frame = imutils.resize(img, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Progress bar update
    progress = 100 * fps_count / total_frame_count
    pbar.update(progress)

    # Loop over the face detections
    for rect in rects:
        # Determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        
        shape = predictor(gray, rect)        
        shape = face_utils.shape_to_np(shape)
        
        # Extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        ears.append((ear, fps_count))

pbar.finish()
print("[INFO] Program complete")
print(ears)