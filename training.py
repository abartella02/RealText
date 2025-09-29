"""
Milestone 2 training file

This model uses a pre-trained word2vec model to obtain weights for words in the dataset.
The word2vec model is a slim version of the Google-News-300 model, available here:
    *  https://github.com/eyaler/word2vec-slim/

References are included in milestone 2 report.
"""

import pandas as pd
import kagglehub
from gensim.models import KeyedVectors
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import tanh, dot, softmax, sum
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM, Embedding, Lambda, Layer
from keras.saving import register_keras_serializable


from sklearn.metrics import confusion_matrix
import spacy

import os
import shutil
from typing import Tuple, Optional
from pathlib import Path
import re
from urllib import request
import json


class Word2Vec:
    """
    Encode a dataframe where the samples are selections of text (essays, sentences, etc)
    using pre-trained word2vec model. Default model is "google-news-300".
    """

    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset
        self.max_len_sentence = self.__get_existing_max_len_sentence()

        model_path = Path(".word2vec/GoogleNews-vectors-negative300-SLIM.bin.gz")

        if not model_path.exists():
            print("downloading pre-trained word2vec model...")
            model_path.parent.mkdir(exist_ok=True)
            download_url = "https://github.com/eyaler/word2vec-slim/raw/refs/heads/master/GoogleNews-vectors-negative300-SLIM.bin.gz?download="
            request.urlretrieve(download_url, model_path)

        # load word2vec model and calculate weights and word indices as normal
        print("Loading model...")
        keyed_vectors = KeyedVectors.load_word2vec_format(str(model_path), binary=True)
        self.weights = keyed_vectors.vectors

        self.word_indices = {
            word.lower(): idx for idx, word in enumerate(keyed_vectors.index_to_key)
        }

    def embed_layer(self, output_dim=None) -> Embedding:
        """Convert word2vec embeddings to tensorflow embedding layer"""
        return Embedding(
            input_dim=self.weights.shape[0],
            output_dim=self.weights.shape[1] if output_dim is None else output_dim,
            weights=[self.weights],
            trainable=False,
        )

    def __get_existing_max_len_sentence(self) -> Optional[int]:
        if not Path('metadata.json').exists():
            return

        with open("metadata.json", "r") as f:
            d = json.load(f)
            if d.get('max_len_sentence', None) is None:
                return
            return d['max_len_sentence']

    def __write_max_len_sentence(self) -> None:
        with open("metadata.json", 'w') as f:
            json.dump({'max_len_sentence':self.max_len_sentence}, f)

    def __words_to_indices(self, sentence: str) -> list[int]:
        """Convert a sentence to a list of indices, readable by a NN's word2vec layer"""
        words = re.findall(r"\w+", sentence)  # split the sentence into a list of words
        return [
            self.word_indices.get(word.lower(), 0) for word in words
        ]  # convert to list of indices

    def encode_text_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode words in dataset as integers corresponding to weights in the embedding layer"""
        encoded_sentences = list(
            self.dataset["Text"].apply(self.__words_to_indices)
        )  # convert to indices


        if self.max_len_sentence is None:
            # should only run during training
            self.max_len_sentence = max(
                len(sentence) for sentence in encoded_sentences
            )  # find sample with most words

            self.__write_max_len_sentence()


        encoded_sentences = pad_sequences(
            encoded_sentences, maxlen=self.max_len_sentence, padding="post"
        )  # pad all other samples

        features = pd.DataFrame(encoded_sentences)  # create feature matrix
        labels = self.dataset["Label"]

        return features, labels


class SamplingStrategy:
    def __init__(self) -> None:
        pass

    def data_cleanup(
        self, dataset: pd.DataFrame, amount_per_class:int, shuffle: bool
    ) -> pd.DataFrame:
        # Drop rows with NaN in the 'text' column
        dataset = dataset.dropna(subset=["text"])

        # Drop any rows from the train data that has emojis
        emoji_rows = dataset[
            dataset["text"].str.contains(r"[\u263a-\U0001f645]", na=False)
        ]
        dataset = dataset.drop(emoji_rows.index, axis=0)

        # Create dataframe for new dataset
        shortened_data = pd.DataFrame()

        # Get an equal amount of random samples from each class and add it to the new dataset df
        # np.random.seed(123)
        if amount_per_class > 0:
            label_1 = dataset[dataset["label"] == 1].sample(
                amount_per_class, random_state=42
            )
            shortened_data = shortened_data._append(label_1)
            label_0 = dataset[dataset["label"] == 0].sample(
                amount_per_class, random_state=42
            )
            shortened_data = shortened_data._append(label_0)
        else:
            shortened_data = dataset

        # Shuffle and reset the index for the new df
        if shuffle:
            shortened_data = shortened_data.sample(frac=1)
        shortened_data = shortened_data.reset_index(drop=True)

        return shortened_data

    def condense_text(self, data: pd.DataFrame, text_column_name: str) -> pd.DataFrame:
        """Function to convert each text sample into single paragraphs with no spacing, new lines, etc"""
        nlp = spacy.load("en_core_web_sm")

        def lemmatize(x):
            nonlocal nlp
            doc = nlp(x)
            return ' '.join([token.lemma_ for token in doc])

        # Replace multiple line breaks with a space for each item in the column
        data[text_column_name] = data[text_column_name].str.replace(
            r"\s*\n\s*", " ", regex=True
        )

        # Normalize spaces (remove multiple spaces)
        data[text_column_name] = (
            data[text_column_name].str.replace(r"\s+", " ", regex=True).str.strip()
        )

        data[text_column_name] = data[text_column_name].apply(lemmatize)
        return data

    def sample_and_clean(
        self,
        data: pd.DataFrame,
        samples_per_class: int,
        shuffle: bool,
        to_csv: bool = False,
        output_csv_path: str = "data/preprocessed.csv",
    ) -> pd.DataFrame:
        """Preprocess using data cleanup and condense text"""

        # shorten the size of the dataset
        train_shortened_data = self.data_cleanup(data, samples_per_class, shuffle=shuffle)

        # get rid of whitespace and other stuff from the texts
        train_no_space_data = self.condense_text(train_shortened_data, "text")

        # rename columns
        train_no_space_data.columns = ["Text", "Label"]

        # put this new train df into a csv
        if to_csv:
            train_no_space_data.to_csv(output_csv_path, index=False)
        return train_no_space_data

@register_keras_serializable(package="Custom", name="AttentionLayer")
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape: tuple[int, int, int]) -> None:
        # input_shape = (batch_size=2, timesteps=5, hidden_dim=3)

        # weights vector to track what features matter most
        # in identifying AI generated words
        self.W = self.add_weight(
            name='att_weight',
            shape=(input_shape[-1], 1),  # input_shape[-1] == number of features
            initializer="glorot_uniform",
            trainable=True
        )

        # positional bias: do words at the beginning matter more than
        # words at the end
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),  # input_shape[1] == sentence length
            initializer="zeros",
            trainable=True
        )

        super().build(input_shape)

    def call(self, x):
        # x := (batch_size=2, timesteps=5, hidden_dim=3)
        imp = tanh(dot(x, self.W) + self.b)  # importance score of each word
        imp_norm = softmax(imp, axis=1)  # normalize
        output = x * imp_norm  # each word scaled by its weight
        return sum(output, axis=1), output


class RNNTextClassifier:
    """
    Basic RNN text classifier, as outlined in Tensorflow's
    "Text classification with an RNN" and "RNN Demo" from AvenueToLearn.
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        w2v_embedding_layer: Optional[Embedding] = None,
    ) -> None:
        self.model = model
        self.embedding_layer = w2v_embedding_layer

    def build(self, input_dimension: int, max_len: int, layer_dim: int = 64) -> None:
        """
        Build the model.
        Layers:
            * Embedding layer (for word2vec)
            * Bidirectional layer
            * Dense layer (fully connected layer with relu activation)

        Use embedded layer from word2vec object if available.
        """
        if isinstance(self.model, Model):
            print("Warning: overwriting pretrained model")

        inputs = Input(shape=(max_len,), name="input")

        if self.embedding_layer is None:
            embed_layer = Embedding(
                input_dim=input_dimension, output_dim=layer_dim, mask_zero=True
            )
        else:
            embed_layer = self.embedding_layer()
        embed_layer_out = embed_layer(inputs)

        bi_lstm_out = Bidirectional(LSTM(embed_layer.output_dim, return_sequences=True))(embed_layer_out)

        attn_layer_out, word_weights = AttentionLayer()(bi_lstm_out)
        # word_weights = Lambda(lambda x: x, name="word_weights")(word_weights)

        relu_layer_out = Dense(embed_layer.output_dim, activation='relu')(attn_layer_out)

        outputs = Dense(1, activation='sigmoid', name='prediction')(relu_layer_out)

        self.model = Model(inputs=inputs, outputs=[outputs, word_weights])

        self.train_model = Model(inputs=inputs, outputs=outputs)

        self.train_model.compile(
            loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"]
        )

    def train(
        self,
        train_text: pd.DataFrame,
        train_label: pd.DataFrame,
        epoch: int,
        batch_size: int,
    ) -> None:

        X = train_text.values if hasattr(train_text, "values") else train_text
        y = train_label.values if hasattr(train_label, "values") else train_label

        # KERAS built-in early stopping, will stop training once loss is the same for two consecutive epochs
        callbacks = [EarlyStopping(monitor = "loss", patience = 2, restore_best_weights = False)]

        self.train_model.fit(
            X, y, epochs=epoch, batch_size=batch_size, verbose=2, callbacks = callbacks
        )

    def predict(self, test_text: pd.DataFrame):

        test_text = tf.convert_to_tensor(test_text.values)
        results, word_weights = self.model(test_text, training=False)
        # results = self.model.predict(test_text, training=False)
        predicted_labels = []
        predicted_confidence = []
        for value in results:
            if value > 0.5:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
            predicted_confidence.append(float(value))

        return predicted_labels, predicted_confidence, word_weights

    def prediction_metrics(
        self, y_predicted: list[int], y_actual: pd.DataFrame
    ) -> Tuple[int, int, int, int]:
        """Return confusion matrix metrics"""
        tn, fp, fn, tp = confusion_matrix(y_actual, y_predicted).ravel()

        return tn, fp, fn, tp


def preprocess(
    dataset: pd.DataFrame, samples_per_class: int = 0, shuffle: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    class Dataset(pd.DataFrame):
        """Subclass of DataFrame to carry on the Word2Vec object as an attribute, for use later"""

        _metadata = ["w2v"]  # Ensures `w2v` persists through Pandas operations

        def __init__(self, data: pd.DataFrame, w2v: Optional[Word2Vec] = None) -> None:
            super().__init__(data)
            self.w2v = w2v

        @property
        def _constructor(self):
            return Dataset

    ss = SamplingStrategy()
    dataset = ss.sample_and_clean(dataset, samples_per_class=samples_per_class, shuffle=shuffle)

    # convert to numerical representation
    _max_len_exists = Path('data.json').exists()
    w2v = Word2Vec(dataset=dataset)
    features, labels = w2v.encode_text_dataset()

    # in case any words are not in our downloaded words
    features = features.clip(0, len(w2v.word_indices) - 1)

    # Combine train_text and train_label for filtering
    data = pd.concat((features, labels), axis=1)

    # Drop rows where train_label is NaN
    data = data.dropna(subset=[labels.name])

    # Separate train_text and train_label
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]

    return Dataset(features, w2v), labels


def download_dataset() -> Path:
    """Download the dataset to ./data/ if the dataset has not already been downloaded"""
    spacy.cli.download('en_core_web_sm')
    # path to dataset in kagglehub
    dataset = "jdragonxherrera/augmented-data-for-llm-detect-ai-generated-text"

    # path to local folder
    dest_path = Path(__file__).parent / "data"

    # if the dataset files we need do not exist locally
    if not (
        (dest_path / "final_test.csv").exists()
        and (dest_path / "final_train.csv").exists()
    ):
        # delete /data/ if it exists but the csv's we need are not inside
        if dest_path.exists():
            shutil.rmtree(dest_path)
        # remake /data/ folder
        dest_path.mkdir(parents=True, exist_ok=True)

        # Download latest version
        # if the kaggle folder is present but no files are inside, the download will not initiate
        path = kagglehub.dataset_download(dataset)
        if len(os.listdir(path)) == 0:
            # if the folder is empty from a previous move operation, delete the folder
            shutil.rmtree(Path(path).parent)
            # attempt to download the dataset again
            path = kagglehub.dataset_download(dataset)

        for item in os.listdir(path):
            # move dataset csv's to repo folder
            shutil.move(os.path.join(path, item), dest_path)

    return dest_path


if __name__ == "__main__":
    import time

    start_time = time.time()
    download_dataset()

    train = pd.read_csv("data/final_train.csv")

    train_features, train_labels = preprocess(train, samples_per_class=15000)

    max_len_sentence = train_features.w2v.max_len_sentence

    with open('data.json', 'w') as f:
        json.dump({"max_len_sentence":max_len_sentence},f)

    rnn = RNNTextClassifier(w2v_embedding_layer=train_features.w2v.embed_layer)

    print("total number of training points")
    print(len(train_labels))

    # input dimension is number of words we have downloaded
    input_dim = len(train_features.w2v.word_indices)
    print(f"Setting input_dim to {input_dim}")
    rnn.build(input_dimension=input_dim, max_len=max_len_sentence)

    # train the model
    print("training...")
    rnn.train(train_features, train_labels, epoch=10, batch_size=10)
    print(f"elapsed = {(time.time()-start_time)//60} minutes")

    print("saving model...")
    # rnn.model.save("temp_model.h5")
    rnn.model.save("temp_model.keras")
