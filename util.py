import numpy as np


def count_a_step(gradient, second_order_differential_in_dot_x):
    """
    Вычисляет динамический шаг градиента (альфа)

    Шаг вычисляется по формуле:

    alpha = (gradient.T * gradient) / (gradient.T * (f''(x) * gradient))
    """

    det = (gradient.dot(second_order_differential_in_dot_x.dot(gradient)))

    if det == 0:
        return 0

    return (gradient.dot(gradient)) / (gradient.dot(second_order_differential_in_dot_x.dot(gradient)))

def transposeVector(vector):
    return vector[:, np.newaxis]
