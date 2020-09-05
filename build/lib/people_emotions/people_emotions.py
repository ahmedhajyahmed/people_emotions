import pandas as pd
import cv2
import os
import csv
import pandas as pd
from people_emotions.emotion_detector.emotion_detector import EmotionDetector
from people_emotions.emotion_detector_fer2013.emotion_detector_fer2013 import EmotionDetectorFer2013
from people_emotions.gender_emotion_detector.gender_emotion_detector import GenderEmotionDetector
PATH = 'C:/Users/ASUS/Desktop/stage/emotion_detection_classifiers/packaging/'


def change_label(label):
    switcher = {
        'fear': 'scared',
        'disgust': 'surprised',
        'surprise': 'surprised'
    }
    return switcher.get(label, label)


def iou(boxA, boxB):
    boxA = boxA[1:-1].split(',')
    boxB = boxB[1:-1].split(',')
    boxA = [int(el) for el in boxA]
    boxB = [int(el) for el in boxB]
    # we compute the intersection coordinates
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # we compute then the intersection area
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # we compute then boxA_area and boxB_area
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # we compute then the union_area
    union_area = boxA_area + boxB_area - inter_area
    return inter_area / union_area


def aggregate_results(results1, results2, results3):
    results1.rename(columns={'bounding box_emotion_detector': 'bb'}, inplace=True)
    results2.rename(columns={'bounding box_emotion_detector_fer2013': 'bb'}, inplace=True)
    results3.rename(columns={'bounding box_gender_emotion_detector': 'bb'}, inplace=True)
    all_frames = pd.concat([results1, results2, results3], ignore_index=True)
    for index1, row1 in all_frames.iterrows():
        for index2, row2 in all_frames.iterrows():
            if index1 < index2:
                value = iou(row1['bb'], row2['bb'])
                if value > 0.7:
                    all_frames.loc[index2, 'bb'] = all_frames.loc[index1, 'bb']
    return all_frames.groupby('bb').first()


class VotingClassifier:

    def __init__(self):
        self.emotion_detector = EmotionDetector()
        self.emotion_detector_fer2013 = EmotionDetectorFer2013()
        self.gender_emotion_detector = GenderEmotionDetector()
        self.labels = ["angry", "scared", "happy", "sad", "surprised", "neutral"]
        self.emotion_detector_weight = 0.49
        self.emotion_detector_fer2013_weight = 0.15
        self.gender_emotion_detector_weight = 0.36
        self.threshold = 0.288  # 0.36 * 80% or 0.49 * 0.59

    def predict(self, image_path, voting_type='soft', results_dir=PATH + 'people_emotions/people_emotions/results'):
        # create the result folder if not exist
        if not (os.path.isdir(results_dir)):
            os.mkdir(results_dir)
        # create the csv file
        base = os.path.basename(image_path)
        file_name = os.path.splitext(base)[0]
        with open(results_dir + '/out_' + file_name + '.csv', 'w', newline='') as file:
            writer = csv.writer(file, dialect='excel', quotechar='"', quoting=csv.QUOTE_ALL, delimiter=',')
            writer.writerow(["bounding box", "final_class", "final_probability"])
        # read the image
        frame = cv2.imread(image_path)
        # load dataframes of all 3 models
        emotion_detector_results = self.emotion_detector.predict(image_path)
        print(emotion_detector_results)
        emotion_detector_fer2013_results = self.emotion_detector_fer2013.predict(image_path)
        print(emotion_detector_fer2013_results)
        gender_emotion_detector_results = self.gender_emotion_detector.predict(image_path)[[
            'bounding box_gender_emotion_detector',
            'emotion_gender_emotion_detector',
            'probEmotion_gender_emotion_detector']]
        print(gender_emotion_detector_results)
        final_results = aggregate_results(emotion_detector_results, emotion_detector_fer2013_results,
                                          gender_emotion_detector_results)

        for index, row in final_results.iterrows():
            score = self.compute_score(final_results, index, voting_type)
            final_class = self.labels[list(score.values()).index(max(list(score.values())))]
            print("la classe est ", final_class)
            final_prob = max(list(score.values()))
            print("avec une prob de ", final_prob)
            if final_prob > self.threshold:
                # write the the bounding box , the class and the probability in the csv file
                x1, y1, x2, y2 = [int(el) for el in index[1:-1].split(',')]
                with open(results_dir + '/out_' + file_name + '.csv', 'a', newline='') as file:
                    writer = csv.writer(file, dialect='excel', quotechar='"', quoting=csv.QUOTE_ALL, delimiter=',')
                    writer.writerow([index, final_class, str(final_prob)])
                cv2.putText(frame, final_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 50, 155), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (140, 50, 155), 2)
        cv2.imwrite(results_dir + "/out_" + base, frame)
        read_file = pd.read_csv(results_dir + '/out_' + file_name + '.csv')
        read_file.to_excel(results_dir + '/out_' + file_name + '.xlsx', index=None, header=True)
        os.remove(results_dir + '/out_' + file_name + '.csv')
        return read_file

    def compute_score(self, final_results, index, voting_type):
        score = {'angry': 0, 'scared': 0, 'happy': 0, 'sad': 0, 'surprised': 0, 'neutral': 0}
        emotion_detector_class = final_results.loc[index, 'class_emotion_detector']
        if not pd.isna(emotion_detector_class):
            emotion_detector_prob = final_results.loc[index, 'probability_emotion_detector']
        # print(emotion_detector_class)
        emotion_detector_fer2013_class = final_results.loc[index, 'class_emotion_detector_fer2013']
        if not pd.isna(emotion_detector_fer2013_class):
            emotion_detector_fer2013_class = change_label(emotion_detector_fer2013_class.lower())
            emotion_detector_fer2013_prob = final_results.loc[index, 'probability_emotion_detector_fer2013']
        # print(emotion_detector_fer2013_class)
        gender_emotion_detector_class = final_results.loc[index, 'emotion_gender_emotion_detector']
        if not pd.isna(gender_emotion_detector_class):
            gender_emotion_detector_class = change_label(gender_emotion_detector_class.lower())
            gender_emotion_detector_prob = final_results.loc[index, 'probEmotion_gender_emotion_detector']
        # print(gender_emotion_detector_class)
        for label in self.labels:
            # print(count)
            # print(label)
            if not pd.isna(emotion_detector_class) and label == emotion_detector_class.lower():
                # print(label, "==", emotion_detector_class)
                if voting_type == 'hard':
                    score[label] += self.emotion_detector_weight
                elif voting_type == 'soft':
                    print(emotion_detector_prob)
                    score[label] += self.emotion_detector_weight * float(emotion_detector_prob)
                    print(score[label])
            if not pd.isna(emotion_detector_fer2013_class) and label == emotion_detector_fer2013_class.lower():
                # print(label, "==", emotion_detector_fer2013_class)
                if voting_type == 'hard':
                    score[label] += self.emotion_detector_fer2013_weight
                elif voting_type == 'soft':
                    print(emotion_detector_fer2013_prob)
                    score[label] += self.emotion_detector_fer2013_weight * float(emotion_detector_fer2013_prob)
                    print(score[label])
            if not pd.isna(gender_emotion_detector_class) and label == gender_emotion_detector_class.lower():
                # print(label, "==", gender_emotion_detector_class)
                if voting_type == 'hard':
                    score[label] += self.gender_emotion_detector_weight
                elif voting_type == 'soft':
                    print(gender_emotion_detector_prob)
                    score[label] += self.gender_emotion_detector_weight * float(gender_emotion_detector_prob)
                    print(score[label])
        print(score)
        return score
