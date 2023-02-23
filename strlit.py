import streamlit as st
from utils import PrepProcesor
import numpy as np
import pandas as pd
from Webcam_2 import DetFace
from Webcam_2 import ImgFace
from fromimg import DetImgFace


import joblib

model = joblib.load('xgbpipe.joblib')

st.title('Hi there! How can I help?')

st.write('Sketch a face for given text input? 👇 ')
st.button('Sketch') 
st.write('Face detection from 👇')

if st.button ('Webcam'):
    DetFace() 

if st.button('Images'):
    # Path = st.text_input('Path to the image')
    # img = ImgFace(Path)
    # st.image(img, caption = 'Face Detected', use_column_width=True)
    # Add an image uploader widget
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        x = ImgFace(image_file)
        print('Hello!')
        st.imshow(x)
        # Show the image
        st.image(image_file, caption='Uploaded Image.', use_column_width=True)
        # Do something with the image
        # ...   
        print("World") 
    else:
        st.write("Please upload an image.")
    
