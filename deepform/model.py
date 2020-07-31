import random
from datetime import datetime

import keras as K
import numpy as np
import tensorflow as tf
from keras.engine.input_layer import Input
from keras.layers import Dense, Dot, Dropout, Flatten, Lambda, concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model

from deepform.common import MODEL_DIR
from deepform.data.add_features import TokenType
from deepform.document import NUM_FEATURES
from deepform.util import git_short_hash


# control the fraction of windows that include a positive label. not efficient.
def one_window(dataset, config):
    require_positive = random.random() > config.positive_fraction
    window = dataset.random_document().random_window(require_positive)
    if config.permute_tokens:
        shuffle = np.random.permutation(config.window_len)
        window.features = window.features[shuffle]
        window.labels = window.labels[shuffle]
    return window


def typed_windows(window):
    """Take a Window and produce a feature vector that pairs it with each TokenType."""
    # F means "Fortran", which just means column-order.
    features = window.features.flatten("F")
    for token_type in TokenType:
        yield np.append(features, [token_type.value]), token_type


def windowed_generator(dataset, config):
    # Create empty arrays to contain batch of features and labels
    # Actual window length is 1 for the active token plus window_len padding on
    # each side. We have NUM_FEATURES for each of these tokens, plus 1 for the
    # type of the active token.
    feature_count = (2 * config.window_len + 1) * NUM_FEATURES + 1
    batch_features = np.zeros((config.batch_size, feature_count))
    batch_labels = np.zeros((config.batch_size))

    while True:
        for i in range(config.batch_size // len(TokenType)):
            window = one_window(dataset, config)
            for j, features_and_token in enumerate(typed_windows(window)):
                features, token_type = features_and_token
                batch_features[i + j, :] = features
                batch_labels[i + j] = window.label == token_type.value
        yield batch_features, batch_labels


# ---- Custom loss function is basically MSE but high penalty for missing a 1 label ---
def missed_token_loss(one_penalty):
    def _missed_token_loss(y_true, y_pred):
        expected_zero = tf.cast(tf.math.equal(y_true, 0), tf.float32)
        s = y_pred * expected_zero
        zero_loss = K.backend.mean(K.backend.square(s))
        expected_one = tf.cast(tf.math.equal(y_true, 1), tf.float32)
        t = one_penalty * (1 - y_pred) * expected_one
        one_loss = K.backend.mean(K.backend.square(t))
        return zero_loss + one_loss

    return _missed_token_loss  # closes over one_penalty


# --- Specify network ---
def create_model(config):

    window_size = 2 * config.window_len + 1
    # Features other than the token id.
    feature_count = window_size * (NUM_FEATURES - 1)

    indata = Input((window_size * NUM_FEATURES + 1,))

    def split_input(x):
        from tensorflow import split

        return split(x, [window_size, feature_count, 1], axis=1)

    tok_str_ids, tok_features, tok_type = Lambda(split_input)(indata)
    id_embed = Flatten()(
        Embedding(config.vocab_size, config.vocab_embed_size)(tok_str_ids)
    )
    tok_embed = Flatten()(Embedding(len(TokenType), config.type_embed_size)(tok_type))
    merged = concatenate([id_embed, tok_features])

    d1 = Dense(
        int(config.window_len * NUM_FEATURES * config.layer_1_size_factor),
        activation="sigmoid",
    )(merged)
    d2 = Dropout(config.dropout)(d1)
    d3 = Dense(
        int(config.window_len * NUM_FEATURES * config.layer_2_size_factor),
        activation="sigmoid",
    )(d2)
    d4 = Dropout(config.dropout)(d3)

    if config.num_layers == 3:
        d5 = Dense(
            int(config.window_len * NUM_FEATURES * config.layer_3_size_factor),
            activation="sigmoid",
        )(d4)
        last_layer = Dropout(config.dropout)(d5)
    else:
        last_layer = d4

    candidate_enc = Dense(config.type_embed_size, activation="elu")(last_layer)
    cosine_sim = Dot(axes=1, normalize=True)([tok_embed, candidate_enc])
    score = K.layers.Activation("sigmoid")(cosine_sim)
    model = Model(inputs=[indata], outputs=[score])

    # _missed_token_loss = missed_token_loss(config.penalize_missed)

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=config.learning_rate),
        # loss=_missed_token_loss,
        loss=K.losses.binary_crossentropy,
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    return model


# --- Predict ---
# Our network is windowed, so we have to aggregate windows to get a final score
# Returns vector of token scores
def predict_scores(model, document, token_type):
    # Here we need to check each label on each window
    flattened = [window.features.flatten("F") for window in document]
    typed = [np.append(features, [token_type.value]) for features in flattened]
    return model.predict(np.stack(typed))


# returns text, score of best answer, plus all scores
def predict_answer(model, document, token_type):
    scores = predict_scores(model, document, token_type)
    best_score_idx = np.argmax(scores)
    best_score_text = document.tokens.iloc[best_score_idx]["token"]
    return best_score_text, scores[best_score_idx], scores


def default_model_name(window_len):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return MODEL_DIR / f"{timestamp}_{git_short_hash()}_{window_len}.model"


def latest_model():
    models = MODEL_DIR.glob("*.model")
    return max(models, key=lambda p: p.stat().st_ctime)


def load_model(model_file=None):
    filepath = model_file or latest_model()
    window_len = int(filepath.stem.split("_")[-1])
    return (
        tf.keras.models.load_model(
            filepath, custom_objects={"_missed_token_loss": missed_token_loss(5)}
        ),
        window_len,
    )


def save_model(model, config):
    basename = config.model_path or default_model_name(config.window_len)
    basename.parent.mkdir(parents=True, exist_ok=True)
    model.save(basename)
