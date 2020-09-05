""" Class that download weights, retrain a CNN model and predict people emotions.

Predicting code adapted from:
    https://github.com/gitshanks/fer2013

Author:
    Ahmed Haj Yahmed (hajyahmedahmed@gmail.com)
"""
# load json and create model
from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import csv
import pandas as pd
import argparse
import numpy as np
import cv2
PATH = 'C:/Users/ASUS/Desktop/stage/emotion_detection_classifiers/packaging/'


class EmotionDetectorFer2013:
    """A class that contains all steps to handel a CNN model for emotion detection.

    This Class download the pre-trained weights, load the model, eventually re-train the model
    and predict people emotions giving a picture.

    Attributes:
        detector (cv.CascadeClassifier): pre_trained haarcasade face detector.
        loaded_model (Keras.model): CNN model for emotion classification.
        labels (list): list of emotion classes.
    """

    def __init__(self, weight_file_path=PATH + "people_emotions/people_emotions/emotion_detector_fer2013/fer.h5",
                 json_file_path=PATH + "people_emotions/people_emotions/emotion_detector_fer2013/fer.json",
                 haarcascade_file_path=PATH + "people_emotions/people_emotions/emotion_detector_fer2013/haarcascade_frontalface_default.xml"):
        """load the face detector cascade, emotion detection CNN,
        then define the list of emotion labels.

        Args:
            weight_file_path (str): path of the CNN model weights.
            json_file_path (str): path of the CNN model.
            haarcascade_file_path (str): path of the haarcascade xml file.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        # loading the model
        json_file = open(json_file_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights(weight_file_path)
        print("Loaded model from disk")
        self.detector = cv2.CascadeClassifier(haarcascade_file_path)
        self.labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def download(self):
        """download the CNN model weights and the haarcascade xml file if not found.
        """
        pass

    def predict(self, image_path, results_dir=PATH + "people_emotions/people_emotions/emotion_detector_fer2013/results"):
        """predict people emotions given an image.

        detect people faces using harrcascade then classify the emotion of the face. This
        method create an output folder containing the input image with the bounding box and the class
        of each face detected and a xlsx file.

        args:
            image_path (str): path of the input image.
            results_dir (str): path of the output folder containing the output image and the output xlsx file.

        returns:
            read_file (pandas.Dataframe) : dataframe containing prediction results.
        """
        # create the result folder if not exist
        if not (os.path.isdir(results_dir)):
            os.mkdir(results_dir)
        # create the csv file
        base = os.path.basename(image_path)
        file_name = os.path.splitext(base)[0]
        with open(results_dir + '/out_' + file_name + '.csv', 'w', newline='') as file:
            writer = csv.writer(file, dialect='excel', quotechar='"', quoting=csv.QUOTE_ALL, delimiter=',')
            writer.writerow(["bounding box_emotion_detector_fer2013", "class_emotion_detector_fer2013",
                             "probability_emotion_detector_fer2013"])

        # setting image resizing parameters
        WIDTH = 48
        HEIGHT = 48
        x = None
        y = None

        # loading image
        full_size_image = cv2.imread(image_path)
        gray = cv2.cvtColor(full_size_image, cv2.COLOR_RGB2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.3, 10)

        # detecting faces
        for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
                cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                yhat= self.loaded_model.predict(cropped_img)
                dominant_prob = np.amax(yhat)
                dominant_class = self.labels[int(np.argmax(yhat))]
                cv2.putText(full_size_image, self.labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                with open(results_dir + '/out_'+file_name+'.csv', 'a', newline='') as file:
                    writer = csv.writer(file, dialect='excel', quotechar='"', quoting=csv.QUOTE_ALL, delimiter=',')
                    writer.writerow(["["+str(x)+","+str(y)+","+str(x + w)+","+str(y + h)+"]",dominant_class, str(dominant_prob)])
        cv2.imwrite(results_dir + "/out_"+base, full_size_image)
        read_file = pd.read_csv(results_dir + '/out_'+file_name+'.csv')
        read_file.to_excel(results_dir + '/out_'+file_name+'.xlsx', index=None, header=True)
        os.remove(results_dir + '/out_'+file_name+'.csv')
        return read_file

    def train(self):
        """re-trained the model for better accuracy.
        """
        pass

