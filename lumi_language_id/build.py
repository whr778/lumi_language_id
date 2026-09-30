import json
from collections import Counter

import langcodes
import numpy as np

from lumi_language_id import LanguageIdentifier, corpus_file, data_file
from lumi_language_id.data_sources import fineweb_gen
from lumi_language_id.tuned import MultiLayerPerceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, log_loss

LENGTH_BUCKETS = ((0, 500, '<500'), (500, 2000, '500-2k'), (2000, float('inf'), '>2k'))

# Languages are weighted by corpus size ** SIZE_EXPONENT, the tempering used for
# multilingual sampling in XLM-R and mT5. Uniform weighting (0) made Cantonese and Wu --
# which fastText labels 'zh' -- 59% of the Han text, so the classifier withheld 42% of
# correct Mandarin answers; at 0.3 it withholds 5.0% and rare languages still count.
SIZE_EXPONENT = 0.3


def language_weights(labels):
    """Per-row weights giving each language size ** SIZE_EXPONENT of the total, mean 1."""
    sizes = json.loads(open(corpus_file('fineweb/sizes.json'), encoding='utf-8').read())
    rows = Counter(labels)
    weights = np.array([sizes[label] ** SIZE_EXPONENT / rows[label] for label in labels])
    return weights / weights.mean()


def get_training_data():
    """
    Training and validation data: FineWeb-2's per-language `train` split, plus FineWeb
    for English, cached locally by `fetch_fineweb`. Every document appears in full and as
    a short window, so the classifier is calibrated on long documents and on short text.

    The original training data (TweetLID and WiLI) held only tweets and Wikipedia
    introductions. The classifier extrapolated past that range on long, spaceless
    Chinese, rejecting fastText's correct answer on 47.7% of Mandarin web documents.
    """
    return make_input_and_output(fineweb_gen('train'))


def get_test_data():
    """
    Held-out test data: FineWeb-2's own `test` split, and a held-out tenth of FineWeb
    for English, prepared the same way as the training data.
    """
    return make_input_and_output(fineweb_gen('test'))


def make_input_and_output(input_gen):
    """
    Convert a generator of labeled examples into the 'X' and 'y' arrays that
    are used to train scikit-learn.
    """
    inputs = []
    outputs = []
    labels = []
    lid = LanguageIdentifier()
    for text, label in input_gen:
        row, detected_lang = lid.make_data_point(text)
        match = langcodes.tag_distance(label, detected_lang) <= 5
        inputs.append(row)
        outputs.append(match)
        labels.append(label)
    return np.array(inputs), np.array(outputs), labels


def make_estimator():
    """
    Make a classifier that estimates the probability of the underlying langID
    model's classification.

    The form of the classifier is a multi-layer perceptron, which I settled on
    after trying multiple different models. An advantage of an MLP is that it
    is trained on log-loss, which rewards correctly calibrated probabilities.

    There are only three input features, so we wouldn't get much advantage from
    a hidden layer size whose dimensionality is much higher than that -- but I
    found that having 8 hidden features is advantageous. My intuition here is
    that ReLU features must output a constant 0 below (or above) their
    intercept, while a sum of two ReLU features has the ability to vary in the
    positive or negative direction, providing another degree of freedom.

    Two hidden layers learned this function better than one, and I saw no
    advantage from having 3 or 4 hidden layers.

    You'd think that using a model-selection tool like AutoML would have been
    the answer here, but AutoML told me to use a random forest classifier,
    which is absolutely wrong when the goal is to output well-tuned
    probabilities.
    """
    return MLPClassifier(
        activation='relu', hidden_layer_sizes=(8, 8), alpha=0.1, max_iter=1000, random_state=0
    )


def bucket_report(name, probabilities, inputs, outputs):
    """
    Print, per length bucket and for Han-heavy vs other text, how often the final answer
    is right (fastText correct AND kept) and how often it is withheld as 'und'.

    A test of short sentences alone could never see a failure on long documents, so the
    readout is split by the cleaned text length and the Han share the classifier sees.
    """
    lengths, han = np.expm1(inputs[:, 0]), np.expm1(inputs[:, 3])
    han_share = han / np.maximum(lengths, 1)
    kept = probabilities >= 0.5
    print(f'{name}:')
    for is_han, script in ((True, "Han-heavy"), (False, "other")):
        for low, high, bucket in LENGTH_BUCKETS:
            mask = ((han_share > 0.3) == is_han) & (lengths >= low) & (lengths < high)
            if mask.any():
                right = (kept & outputs)[mask].mean()
                und = (~kept)[mask].mean()
                print(f'\t{script:9} {bucket:6} n={mask.sum():6}  correct {right:6.1%}  und {und:6.1%}')


def run():
    """
    Build the tuned model that re-estimates the probability of a language
    classification.
    """
    estimator = make_estimator()
    input_train, output_train, labels_train = get_training_data()
    weights = language_weights(labels_train)

    # Split the training source into training and validation. Though
    # we don't have a separate validation step -- it's treated the same as
    # test data -- the purpose here is to show the performance of the classifier
    # on held-out data that is like the training data, while the actual
    # test data is FineWeb's own held-out split.
    input_train, input_val, output_train, output_val, weights_train, weights_val = train_test_split(
        input_train, output_train, weights, test_size=0.2, random_state=0
    )

    # Train the estimator
    estimator.fit(input_train, output_train, sample_weight=weights_train)

    # Determine the log loss of using fastText's original "confidence" value as-is,
    # comparing it against the validation labels. This number is too high.
    orig_confidence = 1. - 2. ** -input_val[:, 1]
    orig_loss = log_loss(output_val, orig_confidence, sample_weight=weights_val)
    print(f'Original log loss: {orig_loss:3.3f}')

    # Use the trained estimator to make new predictions on the validation data.
    predictions = estimator.predict(input_val)
    predictions_p = estimator.predict_proba(input_val)[:, 1]

    # Show the accuracy and log-loss of the new estimator.
    model_accuracy = balanced_accuracy_score(output_val, predictions, sample_weight=weights_val)
    model_loss = log_loss(output_val, predictions_p, sample_weight=weights_val)
    print(f'Validation accuracy: {model_accuracy:3.3f}')
    print(f'Log loss: {model_loss:3.3f}')

    # Get the same statistics for the test data.
    input_test, output_test, labels_test = get_test_data()
    weights_test = language_weights(labels_test)
    predictions = estimator.predict(input_test)
    predictions_p = estimator.predict_proba(input_test)[:, 1]
    model_accuracy = balanced_accuracy_score(output_test, predictions, sample_weight=weights_test)
    model_loss = log_loss(output_test, predictions_p, sample_weight=weights_test)
    print(f'Test accuracy: {model_accuracy:3.3f}')
    print(f'Log loss: {model_loss:3.3f}')

    # The shipped classifier against the new one, on the same held-out rows. A version 1
    # file was trained on raw counts, so it is given raw counts.
    shipped = MultiLayerPerceptron.load(data_file('tuned.npz'), version=1)
    raw_test = input_test.copy()
    raw_test[:, [0, 2, 3]] = np.expm1(raw_test[:, [0, 2, 3]])
    shipped_p = np.array([shipped.probability(row) for row in raw_test])
    bucket_report('shipped tuned.npz', shipped_p, input_test, output_test)
    bucket_report('retrained', predictions_p, input_test, output_test)

    # Show a breakdown of test set accuracy per language
    languages = sorted(set(labels_test))
    for lang in languages:
        lang_filter = [label == lang for label in labels_test]
        lang_accuracy = balanced_accuracy_score(output_test[lang_filter], predictions[lang_filter])
        print(f'\t{lang}\t{lang_accuracy:3.3f}')

    # Extract the trained model and save it in a form that doesn't require
    # scikit-learn to run.
    coefs = estimator.coefs_
    intercepts = estimator.intercepts_
    MultiLayerPerceptron.save(data_file('tuned.npz'), coefs, intercepts)


if __name__ == '__main__':
    run()
