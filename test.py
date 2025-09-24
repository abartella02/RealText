import pandas as pd
from keras.src.random import shuffle
from tensorflow.keras.models import load_model

from training import preprocess
from training import RNNTextClassifier, download_dataset


if __name__ == "__main__":
    download_dataset()
    test = pd.read_csv("data/my_test.csv")
    test_features, test_labels = preprocess(test, shuffle=False)

    model = load_model("saved_model.h5")

    rnn = RNNTextClassifier(model=model)

    # predict
    predicted_labels, prediction_confidence = rnn.predict(test_features)
    actual_labels = test_labels.values.tolist()

    cm = rnn.prediction_metrics(
        y_predicted=predicted_labels, y_actual=test_labels
    )
    # tn, fp, fn, tp = cm.ravel()

    # accuracy = (tp + tn) / (tp + fp + fn + tn)
    # sensitivity = tp / (tp + fn)
    # specificity = tn / (tn + fp)
    #
    # print("test accuracy: ", accuracy)
    # print("test sensitivity: ", sensitivity)
    # print("test specificity: ", specificity)

    print(
        "************************************************************************************"
    )
    for index, row in test.iterrows():
        print(f"** Text **\n {row['text'][:150]}...")
        #print(f"** Analysis **\n{prediction_confidence[int(index)][0]*100:.2f}% chance of being AI generated")
        print(f"** Analysis **\n {"AI Generated" if predicted_labels[int(index)] else "not AI generated"}")
        print(f" chance of being AI generated: {prediction_confidence[int(index)]*100:.2f}%")

        print()
