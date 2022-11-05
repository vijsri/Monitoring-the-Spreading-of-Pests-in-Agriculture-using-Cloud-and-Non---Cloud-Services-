import cv2
import numpy as np
import io
import os
from google.cloud import vision
from google.cloud.vision import types
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="My First Project-100c8e30f20e.json"
font  = cv2.FONT_HERSHEY_SIMPLEX
vidcap = cv2.VideoCapture('paddy.mp4')
count = 0
#search_item = input("Item to be searched : ")

print('Processing...')
print(".")
print(".")
while(True):
    success,img = vidcap.read()
    #print('captured frame count',count)
    
    cv2.imwrite("frame.jpg", img)     # save frame as JPEG file

    if(count%5==0):

        client = vision.ImageAnnotatorClient()
        file_name = os.path.join(
            os.path.dirname(__file__),
            'frame.jpg')
  
        with io.open(file_name, 'rb') as image_file:
            content = image_file.read()
        image = types.Image(content=content)

        # Performs label detection on the image file
        response = client.label_detection(image=image)
        labels = response.label_annotations
        
        print('Labels:')
        for label in labels:
            print(label.description,label.score)
            #search_item = pest;
            if(label.description==u'insect'or label.description==u'leaf'or label.description==u'fauna' or label.description==u'organism' or label.description==u'reptile' or label.description==u'invertebrate'):
                cv2.putText(img,'Unhealthy field' , (25, 200), font, 4, (0, 255, 0), 25, cv2.LINE_AA)
                cv2.namedWindow('Object_Detection',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Object_Detection', 600,600)
                cv2.imshow("Object_Detection", img)
            else:
                print("Healthy field")
            
                cv2.namedWindow('Object_Detection',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Object_Detection', 600,600)
                cv2.imshow("Object_Detection", img)
               
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        
            
    count +=1
