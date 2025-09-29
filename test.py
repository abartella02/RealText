import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.src.random import shuffle
from tensorflow.keras.models import load_model
from sklearn.metrics import ConfusionMatrixDisplay

from training import preprocess
from training import RNNTextClassifier, download_dataset


if __name__ == "__main__":
    download_dataset()
    test = pd.read_csv("data/final_test.csv")
    test_features, test_labels = preprocess(test, samples_per_class=10, shuffle=False)

    model = load_model("temp_model.keras", compile=False)

    rnn = RNNTextClassifier(model=model)

    # predict
    predicted_labels, prediction_confidence, word_weights = rnn.predict(test_features)
    actual_labels = test_labels.values.tolist()

    tn, fp, fn, tp = rnn.prediction_metrics(
        y_predicted=predicted_labels, y_actual=test_labels
    )

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)


    print(
        "************************************************************************************"
    )

    # todo: move this to predict()
    embed_layer = rnn.model.layers[1]
    embed_dim = embed_layer.input_dim
    word_indices = test_features.w2v.word_indices
    index_to_word = np.array([None] * embed_dim)
    for word, idx in word_indices.items():
        if idx < embed_dim:
            index_to_word[idx] = word
    index_to_word[0] = "<PAD>"



    for i in range(len(predicted_labels)):
        text = test.iloc[i]['text'][:150] + '...'
        pred = "AI Generated" if predicted_labels[i] else "not AI generated"
        correct_guess = "correct" if predicted_labels[i] == test_labels[i] else "incorrect"
        confidence_perc = prediction_confidence[i] * 100

        words = index_to_word[test_features.iloc[i].values.astype(int)]
        weights = word_weights[i].numpy().mean(axis=1)
        _word_weights = pd.DataFrame({"word":words, "weight":weights})
        _word_weights = _word_weights[_word_weights["word"] != "<PAD>"]

        # todo: get rid of repeat words
        top_words = _word_weights.sort_values("weight", ascending=False).head(20)

        print(f"** Text **\n {text}")
        print(f"** Analysis **\n {pred} ({correct_guess})")
        print(f" chance of being AI generated: {confidence_perc:.2f}%")
        print(f"top red flag words: {top_words}")
        print()

    print("test accuracy: ", accuracy)
    print("test sensitivity: ", sensitivity)
    print("test specificity: ", specificity)
