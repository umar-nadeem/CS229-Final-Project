# imports
import math
import pandas as pd
from IPython import display
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from tensorflow.python.data import Dataset
from tensorflow.keras import layers

def construct_feature_columns():
    bucketized_prefix = tf.feature_column.categorical_column_with_vocabulary_list(
        key="prefix",
        vocabulary_list=["bazelci/", "examples/", "scripts/", "site/", "src/conditions", "src/java_tools", "src/main", "src/test", "src/tools", "third_party/", "tools/"],
        num_oov_buckets=8)
    print(bucketized_prefix)
    
    bucketized_type = tf.feature_column.categorical_column_with_vocabulary_list(
        key="type",
        vocabulary_list=["JAVA", "C/C++", "Starlark", "python", "HTML/CSS/JS"], num_oov_buckets=5)
    print(bucketized_type)

    prefix_x_type = tf.feature_column.crossed_column(set([bucketized_prefix, bucketized_type]), hash_bucket_size=1000)

    feature_columns = set([
        bucketized_prefix,
        bucketized_type,
        prefix_x_type,
    ])

    return feature_columns


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    # convert pandas data into a dict of np arrays
    features = {key: np.array(value) for key, value in dict(features).items()}

    # construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features, targets))  # beware of limits
    ds = ds.batch(batch_size).repeat(num_epochs)

    # shuffle the data, if specified
    if shuffle:
        ds = ds.shuffle(10000)

    # returns the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

# training linear regression model


def train_model(learning_rate, steps, batch_size, feature_columns, training_examples, training_targets,
                validation_examples, validation_targets):
                
    periods = 10
    steps_per_period = steps / periods

    # create a linear regressor object
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns, optimizer='Ftrl')

    def training_input_fn(): return my_input_fn(training_examples, training_targets['CPUTime'],
                                                batch_size=batch_size)

    def predict_training_input_fn(): return my_input_fn(training_examples, training_targets['CPUTime'],
                                                        num_epochs=1, shuffle=False)

    def predict_validation_input_fn(): return my_input_fn(validation_examples, validation_targets['CPUTime'],
                                                          num_epochs=1, shuffle=False)

    # train the model but do so inside a loop so that we can periodically assess loss metrics
    print('Training model...')
    print('RMSE (on training data):')
    training_rmse = []
    validation_rmse = []

    for period in range(0, periods):
        # train the model, starting from prior state
        linear_regressor.train(
            input_fn=training_input_fn, steps=steps_per_period)

        # take a break and compute predictions
        training_predictions = linear_regressor.predict(
            input_fn=predict_training_input_fn)
        training_predictions = np.array(
            [item['predictions'][0] for item in training_predictions])

        validation_predictions = linear_regressor.predict(
            input_fn=predict_validation_input_fn)
        validation_predictions = np.array(
            [item['predictions'][0] for item in validation_predictions])

        # compute training and validation loss
        
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))

        # occasionally print the current loss
        print(' period %02d: %0.2f' %
              (period, training_root_mean_squared_error))

        # add the loss metrics from this period to our list
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print('Model training finished')

    # output to a graph of loss metris over periods
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title('Root Mean Squared Error vs. Periods')
    plt.tight_layout()
    plt.plot(training_rmse, label='training')
    plt.plot(validation_rmse, label='validation')
    plt.ylim([0, 10000])
    plt.legend()
    plt.show()
    
    return linear_regressor


def test_model(learning_rate, steps, batch_size, feature_columns, testing_examples, testing_targets):
    
    periods = 10
    steps_per_period = steps / periods

    # create a linear regressor object
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns, optimizer='Ftrl')

    def testing_input_fn(): return my_input_fn(testing_examples, testing_targets['CPUTime'],
                                                batch_size=batch_size)

    def predict_testing_input_fn(): return my_input_fn(testing_examples, testing_targets['CPUTime'],
                                                        num_epochs=1, shuffle=False)


    # test the model
    print('Testimg model...')
    print('RMSE (on testing data):')
    testing_rmse = []

    for period in range(0, periods):
        # take a break and compute predictions
        testing_predictions = linear_regressor.predict(
            input_fn=predict_testing_input_fn)
        testing_predictions = np.array(
            [item['predictions'][0] for item in testing_predictions])

        # compute testing loss
        testing_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(testing_predictions, testing_targets))

        # occasionally print the current loss
        print(' period %02d: %0.2f' %
              (period, testing_root_mean_squared_error))

        # add the loss metrics from this period to our list
        testing_rmse.append(testing_root_mean_squared_error)
    print('Model testing finished')

    # output to a graph of loss metris over periods
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title('Root Mean Squared Error vs. Periods')
    plt.tight_layout()
    plt.plot(testing_rmse, label='testing')
    plt.ylim([0, 10000])
    plt.legend()
    plt.show()
    
    return linear_regressor


# main function
if __name__ == '__main__':
    # sort input data
    InputData = pd.read_csv('InputData.csv', sep=',')
    training_examples = InputData.head(300)
    validation_examples = InputData.loc[301:400]
    testing_examples = InputData.tail(100)

    # sort target data
    CPUTimes = pd.read_csv('CPUTimes.csv', sep=',')
    CPUTimes = CPUTimes[["CPUTime"]]
    training_targets = CPUTimes.head(300)
    validation_targets = CPUTimes.loc[301:400]
    testing_targets = CPUTimes.tail(100)

    # visualize sorted input data
    print('Training examples summary:')
    display.display(training_examples.describe())
    print('Validation examples summary:')
    display.display(validation_examples.describe())
    print('Testing examples summary:')
    display.display(testing_examples.describe())

    # visualize sorted target data
    print('Training targets summary:')
    display.display(training_targets.describe())
    print('Validation targets summary:')
    display.display(validation_targets.describe())
    print('Testing targets summary:')
    display.display(testing_targets.describe())

    construct_feature_columns()
    # train a linear regression model
    
    train_model(
        learning_rate=1.0,
        steps=500,
        batch_size=100,
        feature_columns=construct_feature_columns(),
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)

    test_model(
        learning_rate=1.0,
        steps=500,
        batch_size=100,
        feature_columns=construct_feature_columns(),
        testing_examples=testing_examples,
        testing_targets=testing_targets)
