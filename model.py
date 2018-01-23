from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Sequential


# A very simple convolutional neural network model that will be used to predict the probability of the presence of a
# stop sign in a small square image
# Created by brendon-ai, January 2018


# Main function to create model
def get_model(window_size):
    # Hyperbolic tangent activation function
    activation = 'tanh'

    # Initialize the Sequential model
    model = Sequential()

    # Two convolutional layers
    model.add(Conv2D(
        input_shape=(window_size, window_size, 3),
        kernel_size=2,
        filters=8,
        strides=2,
        activation=activation
    ))

    model.add(Conv2D(
        kernel_size=2,
        filters=8,
        strides=2,
        activation=activation
    ))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(16, activation=activation))

    # Sigmoid activation is used for the last layer because its outputs are in the range of 0 to 1
    model.add(Dense(1, activation='sigmoid'))

    # Compile model with Adadelta optimizer
    model.compile(
        loss='mse',
        optimizer='adadelta'
    )

    return model
