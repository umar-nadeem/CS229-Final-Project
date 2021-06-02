# imports
import pandas as pd
from IPython import display
import tensorflow as tf
from matplotlib import pyplot as plt


# global variables


# constructing feature columns
def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])

# training linear regression model
def train_model(learning_rate, steps, batch_size, feature_columns, training_examples, training_targets,
                validation_examples, validation_targets):
    """Trains a linear regression model

    In addition to training, this function also priints training progress information as well as a plot of the training and validation loss over time

    Args:
        learning_rate: A `float`, the learning rate
        steps: A non-zero `int`, the total number of training steps. A training step consists of a forward and backward pass using a single batch
        feature_columns: A `set` specifying the input feature columns to use
        training_examples: A `DataFrame` containing one or more columns from `california_housing_dataframe` to use as input features for training
        training_targets: A `DataFrame` containing exactly one columns from `california_housing_dataframe` to use as target for training
        validation_examples: A `DataFrame` containing exactly one or more columns from `california_housing_dataframe` to use as input features for validation
        validation_targets: A `DataFrame` containing exactly one column from `california_housing_dataframe` to use as target for validation
    Returns:
        A `LinearRegressor` object trained on the training data
    """
    periods = 10
    steps_per_period = steps / periods

    # create a linear regressor object
    my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=my_optimizer)

    training_input_fn = lambda: my_input_fn(training_examples, training_targets['median_house_value'],
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples, training_targets['median_house_value'],
                                                    num_epochs=1, shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets['median_house_value'],
                                                      num_epochs=1, shuffle=False)

    # train the model but do so inside a loop so that we can periodically assess loss metrics
    print('Training model...')
    print('RMSE (on training data):')
    training_rmse = []
    validation_rmse = []

    for period in range(0, periods):
        # train the model, starting from prior state
        linear_regressor.train(input_fn=training_input_fn, steps=steps_per_period)

        # take a break and compute predictions
        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # compute training and validation loss
        training_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))

        # occasionally print the current loss
        print(' period %02d: %0.2f' % (period, training_root_mean_squared_error))

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
    plt.legend()

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

    # train a linear regression model
    train_model(
        learning_rate=1.0,
        steps=500,
        batch_size=100,
        feature_columns=construct_feature_columns(training_examples),
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)