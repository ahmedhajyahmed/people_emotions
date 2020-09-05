""" Class that download weights, retrain a CNN model and predict people emotions.

Predicting code adapted from:
    https://github.com/Amol2709/EMOTION-RECOGITION-USING-KERAS

Author:
    Ahmed Haj Yahmed (hajyahmedahmed@gmail.com)
"""
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import os
import csv
import pandas as pd

PATH = 'C:/Users/ASUS/Desktop/stage/emotion_detection_classifiers/packaging/'


class EmotionDetector:
    """A class that contains all steps to handel a CNN model for emotion detection.

    This Class download the pre-trained weights, load the model, eventually re-train the model
    and predict people emotions giving a picture.

    Attributes:
        detector (cv.CascadeClassifier): pre_trained haarcasade face detector.
        model (Keras.model): CNN model for emotion classification.
        EMOTIONS (list): list of emotion classes.
    """

    def __init__(self,
                 weight_file_path=PATH + "people_emotions/people_emotions/emotion_detector/checkpoints/epoch_75.hdf5",
                 haarcascade_file_path=PATH + "people_emotions/people_emotions/emotion_detector/haarcascade_frontalface_default.xml"):
        """load the face detector cascade, emotion detection CNN,
        then define the list of emotion labels.

        Args:
            weight_file_path (str): path of the CNN model weights.
            haarcascade_file_path (str): path of the haarcascade xml file.
        """
        # load the face detector cascade, emotion detection CNN, then define
        # the list of emotion labels
        self.detector = cv2.CascadeClassifier(haarcascade_file_path)
        self.model = load_model(weight_file_path)
        self.EMOTIONS = ["angry", "scared", "happy", "sad", "surprised", "neutral"]

    def download(self):
        """download the CNN model weights and the haarcascade xml file if not found.
        """
        pass

    def predict(self, image_path, results_dir=PATH + "people_emotions/people_emotions/emotion_detector/results"):
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
            writer.writerow(["bounding box_emotion_detector", "class_emotion_detector", "probability_emotion_detector"])

        frame = cv2.imread(image_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameClone = frame.copy()
        # detect faces in the input frame
        rects = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                               flags=cv2.CASCADE_SCALE_IMAGE)
        # ensure at least one face was found before continuing
        for i in range(0, len(rects)):
            (fX, fY, fW, fH) = rects[i]
            # extract the face ROI from the image, then pre-process
            # it for the network
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            # make a prediction on the ROI, then lookup the class# label
            preds = self.model.predict(roi)[0]
            label = self.EMOTIONS[preds.argmax()]
            dominant_class = ""
            dominant_prob = 0
            # loop over the labels + probabilities
            for (i, (emotion, prob)) in enumerate(zip(self.EMOTIONS, preds)):
                if prob > dominant_prob:
                    dominant_prob = prob
                    dominant_class = emotion
            # write the the bounding box , the class and the probability in the csv file
            with open(results_dir + '/out_' + file_name + '.csv', 'a', newline='') as file:
                writer = csv.writer(file, dialect='excel', quotechar='"', quoting=csv.QUOTE_ALL, delimiter=',')
                writer.writerow(
                    ["[" + str(fX) + "," + str(fY) + "," + str(fX + fW) + "," + str(fY + fH) + "]", dominant_class,
                     str(dominant_prob)])
            cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 50, 155), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (140, 50, 155), 2)
        cv2.imwrite(results_dir + "/out_" + base, frameClone)
        read_file = pd.read_csv(results_dir + '/out_' + file_name + '.csv')
        read_file.to_excel(results_dir + '/out_' + file_name + '.xlsx', index=None, header=True)
        os.remove(results_dir + '/out_' + file_name + '.csv')
        return read_file

    def train(self):
        """re-trained the model for better accuracy.
        """
        pass
