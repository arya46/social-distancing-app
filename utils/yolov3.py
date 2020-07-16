import cv2
import streamlit as st

# Load the network. Because this is cached it will only happen once.
@st.cache(allow_output_mutation=True)
def load_network(config_path, weights_path):
    """
    Method to load the YOLO network
    """
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    output_layer_names = net.getLayerNames()
    output_layer_names = [output_layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layer_names
