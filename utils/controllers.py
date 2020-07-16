import streamlit as st
import numpy as np
import os, urllib, cv2, glob, time

from utils.configs import *
from utils.helper_functions import draw_bbox, mid_point, compute_closest, change_2_red

def detect_image(net, output_layer_names, YOLO_PREDICTION_THRESHOLD, YOLO_IOU_THRESHOLD, YOLO_DISTANCE_THRESHOLD):
    """
    Method to monitor social distancing norms on an image.
    """

    image_path = st.file_uploader("Choose an image: (only .png, .jpg, .jpeg supported)", type=['png', 'jpg', 'jpeg'])
    st.markdown('<p>Need an image? Download <a href="https://i.imgur.com/9pNICWQ.png">here</a>.</p>', unsafe_allow_html=True)
    
    if(image_path != None):
        original_image      = cv2.imdecode(np.fromstring(image_path.read(), np.uint8), 1)
        original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        st.markdown(f"<h3 style='text-align: center;'>Original Image<h3>", unsafe_allow_html=True)
        st.image(original_image, use_column_width=True)

        # Run the YOLO neural net.
        blob = cv2.dnn.blobFromImage(original_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        start = time.time()
        layer_outputs = net.forward(output_layer_names)
        end = time.time()
        print("Frame Prediction Time : {:.6f} seconds".format(end - start))

        # Supress detections in case of too low confidence or too much overlap.
        boxes, confidences, class_IDs = [], [], []
        H, W = original_image.shape[:2]
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > YOLO_PREDICTION_THRESHOLD:
                    box = detection[0:4] * np.array([W, H, W, H])
                    centerX, centerY, width, height = box.astype("int")
                    x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_IDs.append(classID)
        
        idx = [index for index, element in enumerate(class_IDs) if element == 0]
        boxes, confidences, class_IDs = [boxes[i] for i in idx], [confidences[i] for i in idx], [class_IDs[i] for i in idx]
        indices = cv2.dnn.NMSBoxes(boxes, confidences, YOLO_PREDICTION_THRESHOLD, YOLO_IOU_THRESHOLD)
        if(len(indices) > 0):
            boxes = [boxes[i] for i in indices.flatten()]

        mids = mid_point(boxes)
        risky_boxes_index = compute_closest(mids, YOLO_DISTANCE_THRESHOLD)
        image = change_2_red(original_image, boxes, risky_boxes_index)

        image = draw_bbox(image, boxes, risky_boxes_index)

        st.markdown(f"<h3 style='text-align: center;'>Predicted Image<h3>", unsafe_allow_html=True)
        # st.markdown(f"<p style='color: red;'>Violates Social Distancing</p><br>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown(f"<h4> People violating social distancing norms are shown in <span style='color:red;'>red colored boxes</span>.</h4>",  unsafe_allow_html=True)

def detect_video(net, output_layer_names, YOLO_PREDICTION_THRESHOLD, YOLO_IOU_THRESHOLD, YOLO_DISTANCE_THRESHOLD, YOLO_FRAME_SKIP):
    """
    Method to monitor social distancing norms on a video.
    """
    
    video_stream = st.file_uploader("Choose a video: (only .mp4 and .avi supported)", type=['mp4', 'avi'])
    st.markdown('<p>Need a video? Download <a href="https://drive.google.com/file/d/1N_nJRs7gStNrfB1G9y2yPs5voe4Ip4On/view?usp=sharing">here (25MB)</a>.</p>', unsafe_allow_html=True)

    if(video_stream != None):

        temporary_location = "temp/video.mp4"
        my_placeholder = st.empty()

        # delete the uploaded files in previous session
        files = glob.glob('temp/*')
        for f in files:
           os.remove(f)

        with open(temporary_location, 'wb') as out: # Open temporary file as bytes
            out.write(video_stream.read())          # Read bytes into file

        times = []
        counter = 0
        vid = cv2.VideoCapture(temporary_location)

        # ------------ uncomment to write the frame into a video -----------

        # output_path = "temp/result.mp4"
        # by default VideoCapture returns float instead of int
        # width       = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height      = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps         = int(vid.get(cv2.CAP_PROP_FPS))
        # codec       = cv2.VideoWriter_fourcc(*'DIVX')
        # out         = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

        st.markdown(f"<h4> People violating social distancing norms are shown in <span style='color:red;'>red colored boxes</span>.</h4>",  unsafe_allow_html=True)
        while True:

            counter += 1
            _, original_image = vid.read()

            if(counter % YOLO_FRAME_SKIP == 0):

                # Run the YOLO neural net.
                try:
                    blob = cv2.dnn.blobFromImage(original_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                except:
                    break
                net.setInput(blob)

                start = time.time()
                layer_outputs = net.forward(output_layer_names)
                end = time.time()
                print("Frame Prediction Time : {:.6f} seconds".format(end - start))

                # Supress detections in case of too low confidence or too much overlap.
                boxes, confidences, class_IDs = [], [], []
                H, W = original_image.shape[:2]
                for output in layer_outputs:
                    for detection in output:
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]
                        if confidence > YOLO_PREDICTION_THRESHOLD:
                            box = detection[0:4] * np.array([W, H, W, H])
                            centerX, centerY, width, height = box.astype("int")
                            x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            class_IDs.append(classID)
                
                idx = [index for index, element in enumerate(class_IDs) if element == 0]
                boxes, confidences, class_IDs = [boxes[i] for i in idx], [confidences[i] for i in idx], [class_IDs[i] for i in idx]
                indices = cv2.dnn.NMSBoxes(boxes, confidences, YOLO_PREDICTION_THRESHOLD, YOLO_IOU_THRESHOLD)
                if(len(indices) > 0):
                    boxes = [boxes[i] for i in indices.flatten()]

                mids = mid_point(boxes)
                risky_boxes_index = compute_closest(mids, YOLO_DISTANCE_THRESHOLD)
                image = change_2_red(original_image, boxes, risky_boxes_index)

                image = draw_bbox(image, boxes, risky_boxes_index)

                times.append(start-end)
                times = times[-20:]
                ms = sum(times)/len(times)*1000
                fps = 1000 / ms

                image = cv2.putText(image, "{:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

                my_placeholder.image(original_image, use_column_width=True)
                # out.write(image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break  # wait for ESC key to exit

            else:
                pass

        # out.release()
        vid.release()
        cv2.destroyAllWindows()
        print('Exited from process')

        os.remove(temporary_location) ## Delete file when done

def detect_realtime(net, output_layer_names, select_device, YOLO_PREDICTION_THRESHOLD, YOLO_IOU_THRESHOLD, YOLO_DISTANCE_THRESHOLD, YOLO_SLEEP_TIME):
    """
    Method to monitor social distancing norms on a real-time video.
    """

    times=[]
    my_placeholder = st.empty()

    vid = cv2.VideoCapture(select_device)

    # ------------ Uncomment to write the images into a video format -----------
    # by default VideoCapture returns float instead of int
    # width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(vid.get(cv2.CAP_PROP_FPS))
    # codec = cv2.VideoWriter_fourcc(*'DIVX')
    # out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

    # st.markdown(f"<h4> People violating social distancing norms are shown in <span style='color:red;'>red colored boxes</span></h4>",  unsafe_allow_html=True)
    
    while(vid.isOpened()):

        _, frame = vid.read()
        original_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Run the YOLO neural net.
        try:
            blob = cv2.dnn.blobFromImage(original_image, 1 / 255.0, (416, 416), swapRB=False, crop=False)
        except:
            st.write('Could not run YOLO!')
            break
        net.setInput(blob)

        start = time.time()
        layer_outputs = net.forward(output_layer_names)
        end = time.time()
        print("Frame Prediction Time : {:.6f} seconds".format(end - start))

        # Supress detections in case of too low confidence or too much overlap.
        boxes, confidences, class_IDs = [], [], []
        H, W = original_image.shape[:2]
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > YOLO_PREDICTION_THRESHOLD:
                    box = detection[0:4] * np.array([W, H, W, H])
                    centerX, centerY, width, height = box.astype("int")
                    x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_IDs.append(classID)
        
        idx = [index for index, element in enumerate(class_IDs) if element == 0]
        boxes, confidences, class_IDs = [boxes[i] for i in idx], [confidences[i] for i in idx], [class_IDs[i] for i in idx]
        indices = cv2.dnn.NMSBoxes(boxes, confidences, YOLO_PREDICTION_THRESHOLD, YOLO_IOU_THRESHOLD)
        
        if(len(indices) > 0):
            boxes = [boxes[i] for i in indices.flatten()]

        mids = mid_point(boxes)
        risky_boxes_index = compute_closest(mids, YOLO_DISTANCE_THRESHOLD)
        image = change_2_red(original_image, boxes, risky_boxes_index)

        image = draw_bbox(image, boxes, risky_boxes_index)

        my_placeholder.image(image, use_column_width=True)
        # out.write(image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # wait for ESC key to exit

        time.sleep(YOLO_SLEEP_TIME)
        
    # out.release()
    vid.release()
    cv2.destroyAllWindows()
    print('Exited from process')