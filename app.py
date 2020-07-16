import streamlit as st
import pandas as pd
import numpy as np
import os, urllib, cv2

from utils.yolov3 import *
from utils.controllers import *
from utils.helper_functions import *
from utils.configs import *

def main():

    # Download external dependencies.
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    net, output_layer_names = load_network(YOLO_DARKNET_CONFIGS, YOLO_DARKNET_WEIGHTS)

    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Image", "Video", "Real-time"])

    if app_mode == "Show instructions":
        st.markdown(get_file_content_as_string("static/instructions.md"), unsafe_allow_html=True)
        st.sidebar.success('To continue select an action.')

    elif app_mode == "Image":
        st.header('Monitor Social Distancing Norms on still Image:')
        detect_image(net, output_layer_names, YOLO_PREDICTION_THRESHOLD, YOLO_IOU_THRESHOLD, YOLO_DISTANCE_THRESHOLD)

    elif app_mode == "Video":
        st.header('Monitor Social Distancing Norms on Videos:')
        detect_video(net, output_layer_names, YOLO_PREDICTION_THRESHOLD, YOLO_IOU_THRESHOLD, YOLO_DISTANCE_THRESHOLD, YOLO_FRAME_SKIP)

    elif app_mode == "Real-time":
        st.header('Monitor Social Distancing Norms on Real-time Videos:')

        def device_mapper(key):
            if key == 0:
                return "Front Camera"
            else:
                return "Other connected device (if any)"

        select_device = st.selectbox("Choose the input device", [0, 1], index=0, format_func=device_mapper)
        detect_realtime(net, output_layer_names, select_device, YOLO_PREDICTION_THRESHOLD, YOLO_IOU_THRESHOLD, YOLO_DISTANCE_THRESHOLD, YOLO_SLEEP_TIME)

if __name__ == "__main__":
    main()
