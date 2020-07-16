import cv2
import numpy as np
import streamlit as st
from scipy.spatial import distance
import os

# External files to download.
EXTERNAL_DEPENDENCIES = {
    "yolov3.weights": {
        "url": "https://pjreddie.com/media/files/yolov3.weights",
        "size": 248007048
    },
    "yolov3.cfg": {
        "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "size": 8342
    }
}

def download_file(file_path):
    """
    Method to download YOLO weights and configs
    """
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(os.path.join('model_data', file_path)):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(os.path.join('model_data', file_path)) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        print('Downloading Dependencies..')
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(os.path.join('model_data', file_path), "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        print('Downloading Complete!')
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    """
    Method to read markdown file

    Inputs:
        path : path to markdown file

    Returns:
        Returns the markdown file as string
    """

    file = open(path, mode='rb')
    # response = urllib.request.urlopen(url)
    return file.read().decode("utf-8")

def draw_bbox(image, bboxes, skip_indices):   
    """
    Method to draw boxes on a "person"

    Inputs:
        image        : image on which boxes are to be drawn
        bboxes       : all the boxes containing "person"
        skip_indices : list of bboxes indices to skip when coloring
    """
    for i, bbox in enumerate(bboxes):
        if i in skip_indices:
            continue
        else:
            (x1, y1), (x2, y2) = (bbox[0], bbox[1]), (bbox[2], bbox[3])
            image = cv2.rectangle(image, (x1, y1), (x1 + x2, y1 + y2), (0, 0, 255), 2)

    return image

def mid_point(bboxes):
    """
    Method to find the bottom center of every bbox

    Input:
        bboxes: all the boxes containing "person"

    Returns:
        Return a list of bottom center of every bbox
    """

    mid_values = list()
    for i in range(len(bboxes)):
        #get the coordinates
        coor = bboxes[i]
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
        
        #compute bottom center of bbox
        x_mid = int(x1 - ((x1 - (x1 + x2)) / 2))
        y_mid = int(y1 + y2)

        mid_values.append((x_mid, y_mid))
        
    return mid_values

def compute_closest(midpoints, threshold):
    """
    Method to find the boxes that violates social distancing norms

    Inputs:
        midpoints : a list of bottom center (as tuple) for all the boxes containing a "person"
        threshold : threshold value

    Returns:
        A list of bboxes indices that violates social distancing norms
    """

    num = len(midpoints)
    dist = np.zeros((num, num))
    p1 = []
    p2 = []
    for i in range(num-1):
        for j in range(i+1, num):
            if i!=j:
                dst = distance.euclidean(midpoints[i], midpoints[j])
                dist[i][j]=dst
            if(dist[i][j]<=threshold):
                p1.append(i)
                p2.append(j)

    return np.unique(p1 + p2)

def change_2_red(image, bboxes, risky_boxes_index):
    """
    Change the boxes that violates social distancing norms to RED

    Inputs:
        image             : image on which red boxes are to be marked
        bboxes            : all the boxes containing "person"
        risky_boxes_index : list of bboxes indices that are to be colored RED

    Returns:
        Returns the original image after coloring the boxes RED
    """

    for i in risky_boxes_index:
        coor = bboxes[i]
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, (x1, y1), (x1 + x2, y1 + y2), (255, 0, 0), 2)

    return image