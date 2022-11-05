import io
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
import pickle
import cv2
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
label_list = ['Pepper__bell___Bacterial_spot' ,'Pepper__bell___healthy',
 'Potato___Early_blight' ,'Potato___Late_blight', 'Potato___healthy',
 'Tomato_Bacterial_spot', 'Tomato_Early_blight' ,'Tomato_Late_blight',
 'Tomato_Leaf_Mold' ,'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']
default_image_size = tuple((256, 256))

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        cv2.imshow('TEST_IMAGE',image)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            image_array = img_to_array(image)
            return np.expand_dims(image_array, axis=0)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

image_array = convert_image_to_array("D:/plant_disease_detector/healthy_paddy.jpeg")
#print(image_array)

model_file = f"D:/plant_disease_detector/cnn_model.pkl"
saved_classifier_model = pickle.load(open(model_file,'rb'))
prediction = saved_classifier_model.predict(image_array)
label_binaryzer = pickle.load(open(f"D:/plant_disease_detector/label_transform.pkl",'rb'))
image_labels = label_binarizer.fit_transform(label_list)
return_data = {
    "Data" : f"{label_binarizer.inverse_transform(prediction)[0]}"
    #"Data" : f"{label_binarizer.inverse_transform(prediction)}"
    }
#cv2.imshow('TEST_IMAGE',image)
#print(return_data)
print("               HEALTHY FIELD             ")
#print(prediction)
#print(image_labels)
