# CODE FOR RECOGNIZING FACES FROM LIVE WEBCAM FEED (CCTV FOOTAGE SCENARIO)

from imutils import paths
import face_recognition
import pickle
import cv2
import os
import time

 
#get paths of each file in folder named Images
#Images here contains my data(folders of various persons)
images_folder_path = 'Images'
#find path of xml file containing haarcascade file 
# cascPathface = os.path.dirname(
#  cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
cascPathface = r'C:\Users\vishwebh\Desktop\Hackathon\haarcascade_frontalface_default.xml'
# cascPathface = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
# load the known faces and embeddings saved in last file
data = pickle.loads(open('face_enc', "rb").read())
start = time.time()
# DetFace()
# end = time.time()
# print("You ran the code for = {:.2f} seconds".format((end-start)))



def Encode():
    imagePaths = list(paths.list_images(images_folder_path))
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
 

def DetFace():
    print("Streaming started")
    # video_capture.set(3,200)
    # video_capture.set(4,200)
    video_capture = cv2.VideoCapture(0)
    # video_capture.set(cv2.CAP_PROP_FPS,10)
    # loop over frames from the video file stream
    end_1 = time.time()
    print("The camera turned on! Time taken = {:.2f} seconds".format((end_1-start)))
    while True:
        # grab the frame from the threaded video stream
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(60, 60),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
    
        # convert the input frame from BGR to RGB 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(gray, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)
        cv2.imshow("Frame", gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


def ImgFace(image):
    print('Universe')
    # load the harcaascade in the cascade classifier
    # faceCascade = cv2.CascadeClassifier(cascPathface)
    # load the known faces and embeddings saved in last file
    # data = pickle.loads(open('face_enc', "rb").read())
    # Find path to the image you want to detect face and pass it here
    # image = cv2.imread(r'C:\Users\vishwebh\Desktop\Hackathon\Images\Vishwesh\face_0_6117.jpg')
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
        return image
        # cv2.imshow("Frame", image)
        # cv2.waitKey(0)
