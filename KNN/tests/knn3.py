import random
from collections import Counter

import numpy as np
import pandas as pd


class CustomKNN:

    def __init__(self):
        self.accurate_predictions = 0
        self.total_predictions = 0
        self.accuracy = 0.0

    def predict(self, training_data, to_predict, k):
        if len(training_data) >= k:
            print("K can not be smaller than the total voting groups")
            return

        distributions = []
        for group in training_data:
            for features in training_data[group]:
                euclidean_distance = np.linalg.norm(np.array(features) - np.array(to_predict))
                distributions.append([euclidean_distance, group])

        results = [i[1] for i in sorted(distributions)[:k]]
        result = Counter(results).most_common(1)[0][0]
        confidence = Counter(results).most_common(1)[0][1] / k

        return result, confidence

    def predict_label(self, class_nbr, group):
        switcher = {
            1: "Iris-virginica",
            2: "Iris-versicolor",
            3: "Iris-setosa",
        }
        group_label = switcher.get(group, "nothing")

        if class_nbr == 1:
            print('Prediction: Iris-virginica | ', group_label)
        elif class_nbr == 2:
            print('Prediction: Iris-versicolor | ', group_label)
        else:
            print('Prediction: Iris-setosa | ', group_label)

    def test(self, test_set, training_set):
        for group in test_set:
            for data in test_set[group]:
                predicted_class, confidence = self.predict(training_set, data, k=5)
                self.predict_label(predicted_class, group)
                if predicted_class == group:
                    self.accurate_predictions += 1
                # else:
                # print("Wrong classification with confidence " + str(confidence * 100) + " and class " + str(
                #     predicted_class))
                self.total_predictions += 1
        self.accuracy = 100 * (self.accurate_predictions / self.total_predictions)
        print("\nAcurracy :", str(self.accuracy) + "%")


def mod_data(df):
    df.replace('Iris-virginica', 1, inplace=True)
    df.replace('Iris-versicolor', 2, inplace=True)
    df.replace('Iris-setosa', 3, inplace=True)


def main():
    df = pd.read_csv('iris.csv')
    mod_data(df)
    data_set = df.astype(float).values.tolist()

    # Shuffle the data_set
    random.shuffle(data_set)

    # 25% of the available data will be used for testing
    test_size = 0.25

    # The keys of the dict are the classes that the data is classfied into
    training_set = {1: [], 2: [], 3: []}
    test_set = {1: [], 2: [], 3: []}

    # Split data into training and test for cross validation
    training_data = data_set[:-int(test_size * len(data_set))]
    test_data = data_set[-int(test_size * len(data_set)):]

    # Insert data into the training set
    for record in training_data:
        training_set[record[-1]].append(
            record[:-1])  # Append the list in the dict will all the elements of the record except the class

    # Insert data into the test set
    for record in test_data:
        test_set[record[-1]].append(
            record[:-1])  # Append the list in the dict will all the elements of the record except the class

    # s = time.clock()
    knn = CustomKNN()
    knn.test(test_set, training_set)
    # e = time.clock()
    # print("Exec Time:", e - s)


if __name__ == "__main__":
    main()
