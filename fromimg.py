# Extracting features from face
from imutils import paths
import face_recognition
import pickle
import cv2
import os

def Enco():
    #Get paths of each file in folder named Images
    #Images here contains my data(folders of various persons)
    imagePaths = list(paths.list_images('Images'))
    knownEncodings = []
    knownNames = []
    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        name = imagePath.split(os.path.sep)[-2]
        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Use Face_recognition to locate faces
        boxes = face_recognition.face_locations(rgb,model='hog')
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
        # loop over the encodings
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)
    #save emcodings along with their names in dictionary data
    data = {"encodings": knownEncodings, "names": knownNames}
    #use pickle to save data into a file for later use
    f = open("face_enc", "wb")
    f.write(pickle.dumps(data))
    f.close()

#find path of xml file containing haarcascade file
cascPathface = r'C:\Users\vishwebh\Desktop\Hackathon\haarcascade_frontalface_default.xml'


def DetImgFace():
    # load the harcaascade in the cascade classifier
    faceCascade = cv2.CascadeClassifier(cascPathface)
    # load the known faces and embeddings saved in last file
    data = pickle.loads(open('face_enc', "rb").read())
    #Find path to the image you want to detect face and pass it here
    image = cv2.imread(r'C:\Users\vishwebh\Desktop\Hackathon\Images\Vishwesh\face_0_6117.jpg')
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #convert image to Greyscale for haarcascade
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)
        else: # To store the unknown new face with name
            faces = faceCascade.detectMultiScale(gray,
                                    scaleFactor=1.1,
                                    minNeighbors=5,
                                    minSize=(60, 60),
                                    flags=cv2.CASCADE_SCALE_IMAGE)
                                    
        cv2.imshow("Frame", image)
        cv2.waitKey(0)