import sys
import os
import argparse
import csv
import cv2
import pandas as pd
from keras.models import load_model
import numpy as np

from people_emotions.gender_emotion_detector.utils.datasets import get_labels
from people_emotions.gender_emotion_detector.utils.inference import detect_faces
from people_emotions.gender_emotion_detector.utils.inference import draw_text
from people_emotions.gender_emotion_detector.utils.inference import draw_bounding_box
from people_emotions.gender_emotion_detector.utils.inference import apply_offsets
from people_emotions.gender_emotion_detector.utils.inference import load_detection_model
from people_emotions.gender_emotion_detector.utils.inference import load_image
from people_emotions.gender_emotion_detector.utils.preprocessor import preprocess_input
PATH = 'C:/Users/ASUS/Desktop/stage/emotion_detection_classifiers/packaging/'


class GenderEmotionDetector:

    def __init__(self, detection_model_path=PATH + "people_emotions/people_emotions/gender_emotion_detector/trained model/haarcascade_frontalface_default.xml",
                 emotion_model_path=PATH + "people_emotions/people_emotions/gender_emotion_detector/trained model/fer2013_mini_XCEPTION.102-0.66.hdf5",
                 gender_model_path=PATH + "people_emotions/people_emotions/gender_emotion_detector/trained model/simple_CNN.81-0.96.hdf5", emotion_labels_id="fer2013",
                 gender_labels_id="imdb"):
        # loading models
        self.face_detection = load_detection_model(detection_model_path)
        self.emotion_classifier = load_model(emotion_model_path, compile=False)
        self.gender_classifier = load_model(gender_model_path, compile=False)
        # parameters for loading data
        self.emotion_labels = get_labels(emotion_labels_id)
        self.gender_labels = get_labels(gender_labels_id)

    def download(self):
        pass

    def predict(self, image_path, results_dir=PATH + "people_emotions/people_emotions/gender_emotion_detector/results"):
        # create the result folder if not exist
        if not (os.path.isdir(results_dir)):
            os.mkdir(results_dir)
        # create the csv file
        base = os.path.basename(image_path)
        file_name = os.path.splitext(base)[0]
        with open(results_dir + '/out_' + file_name + '.csv', 'w', newline='') as file:
            writer = csv.writer(file, dialect='excel', quotechar='"', quoting=csv.QUOTE_ALL, delimiter=',')
            writer.writerow(["bounding box_gender_emotion_detector", "emotion_gender_emotion_detector",
                             "probEmotion_gender_emotion_detector", "gender_gender_emotion_detector",
                             "probGender_gender_emotion_detector"])

        # hyper-parameters for bounding boxes shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        # gender_offsets = (30, 60)
        gender_offsets = (10, 10)
        # emotion_offsets = (20, 40)
        emotion_offsets = (0, 0)

        # getting input model shapes for inference
        emotion_target_size = self.emotion_classifier.input_shape[1:3]
        gender_target_size = self.gender_classifier.input_shape[1:3]

        # loading images
        rgb_image = load_image(image_path, grayscale=False)
        gray_image = load_image(image_path, grayscale=True)
        gray_image = np.squeeze(gray_image)
        gray_image = gray_image.astype('uint8')

        faces = detect_faces(self.face_detection, gray_image)
        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
            rgb_face = rgb_image[y1:y2, x1:x2]

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]

            try:
                rgb_face = cv2.resize(rgb_face, (gender_target_size))
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            # preprocess the image before passing it to the classifier
            rgb_face = preprocess_input(rgb_face, False)
            rgb_face = np.expand_dims(rgb_face, 0)
            # predict gender and getting the gender class and its probability
            gender_prediction = self.gender_classifier.predict(rgb_face)
            gender_prob = np.amax(gender_prediction)
            gender_label_arg = np.argmax(gender_prediction)
            gender_text = self.gender_labels[gender_label_arg]

            # preprocess the image before passing it to the classifier
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            # predict gender and getting the gender class and its probability
            emotion_prediction = self.emotion_classifier.predict(gray_face)
            emotion_prob = np.amax(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = self.emotion_labels[emotion_label_arg]
            fX, fY, fW, fH = face_coordinates
            # writing the bounding box mesures, the classes and their probabilities in the csv file
            with open(results_dir + '/out_' + file_name + '.csv', 'a', newline='') as file:
                writer = csv.writer(file, dialect='excel', quotechar='"', quoting=csv.QUOTE_ALL, delimiter=',')
                writer.writerow(
                    ["[" + str(fX) + "," + str(fY) + "," + str(fX + fW) + "," + str(fY + fH) + "]", emotion_text,
                     str(emotion_prob), gender_text, str(gender_prob)])

            if gender_text == self.gender_labels[0]:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, gender_text, color, 60, 0, 0.7, 2)
            draw_text(face_coordinates, rgb_image, emotion_text, color, -20, 0, 0.7, 2)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(results_dir + "/out_" + base, bgr_image)
        read_file = pd.read_csv(results_dir + '/out_' + file_name + '.csv')
        read_file.to_excel(results_dir + '/out_' + file_name + '.xlsx', index=None, header=True)
        os.remove(results_dir + '/out_' + file_name + '.csv')
        return read_file

    def train(self):
        pass