import warnings

import pandas as pd

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score

import mlflow.sklearn


def loadData():
    warnings.filterwarnings('ignore')

    v_browser = pd.read_csv('../data/v_browser.csv')
    v_platform = pd.read_csv('../data/v_platform.csv')
    return v_browser, v_platform


def encode_labels_browser(df):
    date_encoder = preprocessing.LabelEncoder()
    device_encoder = preprocessing.LabelEncoder()
    browser_encoder = preprocessing.LabelEncoder()
    experiment_encoder = preprocessing.LabelEncoder()
    target_encoder = preprocessing.LabelEncoder()

    df['date'] = date_encoder.fit_transform(df['date'])
    df['device_make'] = device_encoder.fit_transform(df['device_make'])
    df['browser'] = browser_encoder.fit_transform(df['browser'])
    df['experiment'] = experiment_encoder.fit_transform(df['experiment'])
    df['target'] = target_encoder.fit_transform(df['target'])

    return df, date_encoder, device_encoder, browser_encoder, experiment_encoder, target_encoder


def train_test_val_split(X, Y, split=(0.2, 0.1), shuffle=True):
    """Split dataset into train/val/test subsets by 70:20:10(default).

    Args:
      X: List of data.
      Y: List of labels corresponding to data.
      split: Tuple of split ratio in `test:val` order.
      shuffle: Bool of shuffle or not.

    Returns:
      Three dataset in `train:test:val` order.
    """
    from sklearn.model_selection import train_test_split
    assert len(X) == len(Y), 'The length of X and Y must be consistent.'
    X_train, X_test_val, y_train, Y_test_val = train_test_split(X, Y,
                                                                test_size=(split[0] + split[1]), shuffle=shuffle)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, Y_test_val,
                                                    test_size=split[1], shuffle=False)
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


def test(self, x_test, y_test):
    y_pred = self.clf.predict(x_test)
    accuracy = self.calculate_score(y_test, y_pred)
    self.__printAccuracy(accuracy, label="Test")
    report = self.report(y_pred, y_test)
    matrix = self.confusion_matrix(y_pred, y_test)
    return accuracy, report, matrix


if __name__ == '__main__':
    mlflow.set_experiment(experiment_name='ML_ABtest')
    print("loading data")
    v_browser, v_platform = loadData()
    encoded_df, date_encoder, device_encoder, browser_encoder, experiment_encoder, target_encoder = encode_labels_browser(
        v_browser)
    feature_col = ["experiment", "hour", "date", "device_make", "browser"]
    features_X = encoded_df[feature_col]
    target_y = encoded_df["target"]
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = train_test_val_split(features_X, target_y)
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    # create model
    model_logistic = LogisticRegression()

    # Train model
    print("Training")

    model_logistic.fit(X_train, y_train)
    print("Training ended")
    threshold = 0.5
    print("Predicting")
    y_pred_logistic = model_logistic.predict(X_test)
    mlflow.log_param("Logistic_Predictions", y_pred_logistic)

    # evaluate model using cross validation
    scores = cross_val_score(model_logistic, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)

    logistic_accuracy = accuracy_score(y_test, y_pred_logistic)

    print(logistic_accuracy)
    mlflow.log_metric("logistic_accuracy", logistic_accuracy)
    mlflow.sklearn.log_model(model_logistic, "Logistic Model")
