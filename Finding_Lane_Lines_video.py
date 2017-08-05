from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import imageio
imageio.plugins.ffmpeg.download()
from scipy.signal import argrelmax
from scipy.signal import find_peaks_cwt

from CameraCalibrator import CameraCalibrator
from PerspectiveTransformer import PerspectiveTransformer
from LaneFinder import LaneFinder

# Camera calibration constants
CAMERA_CAL_IMAGES_PATH = "./camera_cal"
CAMERA_CAL_IMAGE_NAMES = "calibration*.jpg"
CAMERA_CAL_PICKLE_NAME = "calibration_data.p"



def process_video(video_name):
    '''Detect lane lines in an entire video and write the result to disc'''
    lf = LaneFinder(calibrator, perspective_transformer, n_frames=7)

    video_input = VideoFileClip(video_name + ".mp4")
    video_output = video_name + "_output.mp4"
    output = video_input.fl_image(lf.process_video_frame)
    output.write_videofile(video_output, audio=False)

# Calibrate the camera - one-off
calibrator = CameraCalibrator(CAMERA_CAL_IMAGES_PATH, CAMERA_CAL_IMAGE_NAMES, CAMERA_CAL_PICKLE_NAME)

# Create the perspective transform - one-off (assumption: the road is a flat plane)
perspective_transformer = PerspectiveTransformer()

# Process a video
process_video("project_video")
