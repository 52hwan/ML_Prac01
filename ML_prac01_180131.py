# TensorFlow tutorials
#
# https://www.tensorflow.org/get_started/premade_estimators
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import argparse


CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Sentosa', 'Versicolor', 'Virginica']

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


def load_data(y_name='Species'):
    """ Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path = 'iris_training.csv' # We don't need to download.

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test_path = 'iris_test.csv'

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """ An input function for training """
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Build the Iterator, and return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels, batch_size):
    """ An input function for evaluating or prediction"""
    features = dict(features)

    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data.
    (train_x, train_y), (test_x, test_y) = load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    my_checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs=20 * 60,  # Save checkpoints every 20 minutes
        keep_checkpoint_max=10  # Retain the 10 most recent checkpoints
    )

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns,
                                            hidden_units=[10, 10],
                                            n_classes=3,
                                            model_dir='models/iris',
                                            config=my_checkpointing_config)

    # Train the model.
    classifier.train(input_fn=lambda:train_input_fn(train_x, train_y, args.batch_size),
                     steps=args.train_steps)
    # Without 'lambda', not working. The function 'train_input_fn' returns different results when iterated

    # Evaluate the model.
    eval_result = classifier.evaluate(input_fn=lambda:eval_input_fn(test_x, test_y,
                                                                    args.batch_size))

    print('\n Test set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda:eval_input_fn(predict_x, labels=None, batch_size=args.batch_size))

    for pred_dict, expec in zip(predictions, expected):

        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)