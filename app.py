import streamlit as st
from imutils import paths
import face_recognition
import pickle
import numpy as np
from PIL import Image
import cv2
from Webcam_2 import DetFace
from serpapi import GoogleSearch

cascPathface = r'C:\Users\vishwebh\Desktop\Hackathon\haarcascade_frontalface_default.xml'

st.title('Hi there! How can I help?')

st.write('Sketch a face for given text input? 👇 ')
st.button('Sketch') 
st.write('Face recognition from Webcam👇')

if st.button ('Webcam'):
    DetFace() 

st.write('Face recognition using images👇')
# if st.button('Images'):

faceCascade = cv2.CascadeClassifier(cascPathface)
# st.write('Detection model loaded...')
data = pickle.loads(open('face_enc', "rb").read())
# st.write('Encodings loaded...')
image_upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
# image = Image.open(image_upload)
# img_array = np.array(image)
   
if image_upload is not None:
    file_bytes = np.asarray(bytearray(image_upload.read()), dtype=np.uint8) # https://github.com/streamlit/streamlit/issues/888
    opencv_image = cv2.imdecode(file_bytes, 1)
    # st.image(image, caption=f"Uploaded Image {img_array.shape[0:2]}", use_column_width=True,)
    # image = cv2.imread(img_array)
    rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    #convert image to Greyscale for haarcascade
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(60, 60),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
    
    # the facial embeddings for face in input
    encodings = face_recognition.face_encodings(rgb)
    names = []
    # loop over the facial embeddings incase
    # we have multiple embeddings for multiple fcaes
    for encoding in encodings:
        #Compare encodings with encodings in data["encodings"]
        #Matches contain array with boolean values and True for the embeddings it matches closely
        #and False for rest
        matches = face_recognition.compare_faces(data["encodings"],
        encoding)
        #set name =inknown if no encoding matches
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            #Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                #Check the names at respective indexes we stored in matchedIdxs
                name = data["names"][i]
                #increase count for the name we got
                counts[name] = counts.get(name, 0) + 1
                #set name which has highest count
                name = max(counts, key=counts.get)
            # update the list of names
            names.append(name)
            # loop over the recognized faces
            for ((x, y, w, h), name) in zip(faces, names):
                # rescale the face coordinates
                # draw the predicted face name on the image
                cv2.rectangle(opencv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(opencv_image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)
        else: # To store the unknown new face with name
            faces = faceCascade.detectMultiScale(gray,
                                    scaleFactor=1.1,
                                    minNeighbors=5,
                                    minSize=(60, 60),
                                    flags=cv2.CASCADE_SCALE_IMAGE)
        
        # cv2.imwrite('Hola.png',image)
        st.write('This is' + name)
        st.image(opencv_image, caption='Detection Result', use_column_width=True) 
# else:
#     st.write("Please upload an image.")

st.write('For image websearch👇')
image_upload2 = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if image_upload2 is not None:
    params = {
    "engine": "google_lens",
    "q": "Coffee",
    "url": "https://i.imgur.com/HBrB8p0.png",
    "hl": "en",
    "api_key": "secret_api_key"
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    
