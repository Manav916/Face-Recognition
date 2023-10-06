import numpy as np
import cv2
import sklearn
import pickle
import tensorflow as tf
from keras.preprocessing import image
from django.conf import settings
import os

STATIC_DIR = settings.STATIC_DIR


# Face detection
face_detector_model = cv2.dnn.readNetFromCaffe(os.path.join(STATIC_DIR,'models/deploy.prototxt.txt'), os.path.join(STATIC_DIR, 'models/res10_300x300_ssd_iter_140000_fp16.caffemodel'))
# Feature extraction
face_feature_model = cv2.dnn.readNetFromTorch(os.path.join(STATIC_DIR, 'models/openface.nn4.small2.v1.t7'))
# Face Recognition
face_recognition_model = pickle.load(open(os.path.join(STATIC_DIR,'models/machine_learning_face_person_identity.pkl'), mode='rb'))
# Emotion Recognition
emotion_recognition_model = tf.keras.models.load_model(os.path.join(STATIC_DIR,"models/emotion_model_cnn.h5"))
# emotion_recognition_model = pickle.load(open('./models/machine_learning_face_emotion.pkl', mode='rb'))



emotions = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'neutral', 5:'sad', 6:'suprise'}

# Pipeline model
def pipeline(path):
    # Reading images
    
    img = cv2.imread(path)
    image = img.copy()
    h,w = img.shape[:2]
    # Face detection 
    
    img_blob = cv2.dnn.blobFromImage(img, 1, (300,300), (104,177,123), swapRB=False, crop=False)
    face_detector_model.setInput(img_blob)
    detections = face_detector_model.forward()
    
    # Results
    results = dict(face_detect_score = [], 
                             face_name = [],
                             face_name_score = [],
                             emotion_name = [],
                             emotion_name_score = [],
                             count = [])
    
    count = 1
    if len(detections)>0:
        for i, confidence in enumerate(detections[0,0,:,2]):
            if confidence>0.5:
                box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                startx, starty, endx, endy = box.astype(int)
                
                cv2.rectangle(image, (startx,starty), (endx,endy), (0,255,0))
                
                # Feature extraction
                face_roi = img[starty:endy,startx:endx]
                face_blob = cv2.dnn.blobFromImage(face_roi,1/255,(96,96),(0,0,0),swapRB=True,crop=True)
                face_feature_model.setInput(face_blob)
                vectors = face_feature_model.forward()
                
                # Using saved machine learning model to predict
                face_name = face_recognition_model.predict(vectors)[0]
                face_score = face_recognition_model.predict_proba(vectors).max()
                
                # Emotion prediction using cnn model
                # Emotion image preprocessing 
                emotion_image = cv2.resize(face_roi, (64,64))
                emotion_image = emotion_image/255.
                emotion_image = np.expand_dims(emotion_image, axis = 0)
                emotion_result = emotion_recognition_model.predict(emotion_image)
                emotion_name = emotions[emotion_result.argmax()]                
                emotion_score = round(emotion_result.max(), 2)
                
                text_face = '{} : {:.0f} %'.format(face_name,100*face_score)
                text_emotion = '{} : {:.0f} %'.format(emotion_name,emotion_score)
                cv2.putText(image,text_face,(startx,starty),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
                cv2.putText(image,text_emotion,(startx,endy),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT, 'ml_output/process.jpg'),image)
                cv2.imwrite(os.path.join(settings.MEDIA_ROOT, 'ml_output/roi_{}.jpg'.format(count)),face_roi)
                
                
                results['count'].append(count)
                results['face_detect_score'].append(round(confidence, 2))
                results['face_name'].append(face_name)
                results['face_name_score'].append(round(face_score, 2))
                results['emotion_name'].append(emotion_name)
                results['emotion_name_score'].append(emotion_score)
                
                count += 1
                
            
    return results



