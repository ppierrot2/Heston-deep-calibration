import numpy as np
from keras import backend as K


def monte_carlo_prediction(model, X, forward_path=100):
    """
    take a fitted keras model implementing dropout
    and generate forward path into the network resulting
     in different prediction for X

    :param model: keras model or sequential
    :param X: input data
    :param forward_path: number of prediction path to generate
    :return: array
    """
    MC_output = K.function([model.layers[0].input, K.learning_phase()],
                           [model.layers[-1].output])
    learning_phase = True  # use dropout at test time

    MC_samples = [MC_output([X, learning_phase])[0] for _ in range(forward_path)]
    MC_samples = np.array(MC_samples)

    return MC_samples


def compute_uncertainty(samples_prediction):
    """
    compute MC mean, var and confidence

    :param samples_prediction: array of mc prediction path
    :return: tuple : (mean, std, conf_inf 30%, conf_sup 70%, conf_inf 10%, conf_sup 90%,)
    """

    mean = np.mean(samples_prediction, axis=0)
    std = np.std(samples_prediction, axis=0)
    return mean, std, mean - std, mean + std, mean - 1.96 * std, mean + 1.96 * std

