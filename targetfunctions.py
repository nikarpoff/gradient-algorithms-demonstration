import numpy as np


class TargetFunction:
    """
    Данный класс представляет обобщение для всех многомерных функций.

    Класс оперирует вектором x, где каждый элемент вектора - переменная, по которой строится функция.

    Для методов второго порядка определена функция secondOrderDifferentialInDot, возвращающая значение
    второй производной функции в точке с координатами x.
    """

    def __call__(self, x):
        pass

    def differential_in_dot(self, x):
        """
        Для применения градиентных методов в классах, унаследованных от TargetFunction, должен
        быть определен метод differentialInDot, возвращающий значение производной функции в точке с координатами x.
        """
        pass

    def second_order_differential_in_dot(self, x):
        """
        Для применения методов второго порядка в классах, унаследованных от TargetFunction, должен быть определен 
        метод secondOrderDifferentialInDot, возвращающий значение второй производной функции в точке с координатами x.
        """
        pass


class EllipticalParaboloid(TargetFunction):
    """
    Данный класс представляет эллиптический параболоид заданный в общем виде:
        f(x) = xT * A * x - bT * x + c
    где A - квадратная матрица размером nxn, x - вектор переменных размером 1xn, b и c - вектора размером 1xn
    """

    def __init__(self, A, b, c):
        self.b = np.array(b)
        self.c = c
        self.A = np.array(A)

    def __call__(self, x):
        return (1 / 2) * np.array(x).T.dot(self.A).dot(x) - self.b.dot(x) + self.c

    # differential is: xT * A - b
    def differential_in_dot(self, x):
        return np.dot(x, self.A) - self.b

    # second order differential is: A
    def second_order_differential_in_dot(self, x):
        return self.A


class OscillatoryBivariateFunction(TargetFunction):
    """
    Данный класс представляет двумерную осциллирующую функцию:
        3 * sin(x0) + 2 * cos(x1)
    где x - вектор переменных, в котором учитываются только первая и вторая (x0 и x1)
    """

    def __call__(self, x):
        return 2 * np.cos(x[1]) + 3 * np.sin(x[0])

    # differential in x0: 3 * cos(x0); in x1: - 2 * sin(x1)
    def differential_in_dot(self, x):
        return np.array([3 * np.cos(x[0]), - 2 * np.sin(x[1])])

    # second order differential in x0: [3, 0]; in x1: [0, 2]
    def second_order_differential_in_dot(self, x):
        return np.array([[3, 0], [0, 2]])


def count_squares_sum(x):
    return x[0] * x[0] + x[1] * x[1]


# class UnimodalPitFunction(TargetFunction):
#     """
#     Данный класс представляет двумерную функцию вида:
#         - (x0^2 + x1^2)^2 + 2 * (x0^2 + x1^2) + 2
#     где x - вектор переменных, в котором учитываются только первая и вторая (x0 и x1)
#     """
#     def __call__(self, x):
#         return -np.power(count_squares_sum(x), 2) + 2 * count_squares_sum(x) + 2
#
#         # differential in x0: -4x0 * (x0^2 + x1^2 - 1); in x1: -4x1 * (x0^2 + x1^2 - 1)
#     def differential_in_dot(self, x):
#         return np.array([-4 * x[0] * (count_squares_sum(x) - 1),
#                          -4 * x[1] * (count_squares_sum(x) - 1)])
#
#         # second order differential in x0: -4 * (x0^2 + x1^2 - 1) - 4x0^2;
#         #                           in x1: -4 * (x0^2 + x1^2 - 1) - 4x1^2
#         # second order differential in x0: [-12x0^2 - 4x1^2 + 4, -8x1*x0]; in x1: [-8x1*x0, -12x1^2 - 4x0^2 + 4]
#     def second_order_differential_in_dot(self, x):
#         # return np.array([[-12 * x[0] * x[0] - 4 * x[1] * x[1] + 4, -8 * x[0] * x[1]],
#         #                 [-8 * x[0] * x[1], -12 * x[1] * x[1] - 4 * x[0] * x[0] + 4]])
#
#         return np.array([[-12 * x[0] * x[0] - 4 * x[1] * x[1] + 4, -8 * x[0] * x[1]],
#                          [-8 * x[0] * x[1], -12 * x[1] * x[1] - 4 * x[0] * x[0] + 4]])
#
#         # return np.array([[-4 * (count_squares_sum(x) - 1) - 8 * x[0] ** 2, 0],
#         #                  [0, -4 * (count_squares_sum(x) - 1) - 8 * x[1] ** 2]])

class UnimodalPitFunction(TargetFunction):
    """
    Данный класс представляет двумерную функцию вида:
        - (x0^2 + x1^2)^2 + 2 * (x0^2 + x1^2) + 2
    где x - вектор переменных, в котором учитываются только первая и вторая (x0 и x1)
    """

    def __call__(self, x):
        return -(x[0]**2 + x[1]**2)**2 + 2 * (x[0]**2 + x[1]**2) + 2

    # differential in x0: -4 * x0 * (x0^2 + x1^2) + 4 * x0;
    # in x1: -4 * x1 * (x0^2 + x1^2) + 4 * x1
    def differential_in_dot(self, x):
        return np.array([-4 * x[0] * (x[0]**2 + x[1]**2) + 4 * x[0],
                         -4 * x[1] * (x[0]**2 + x[1]**2) + 4 * x[1]])

    # second order differential in x0: [-12 * x0^2 + 12 * (x0^2 + x1^2) + 4, -8 * x0 * x1];
    # in x1: [-8 * x0 * x1, -12 * x1^2 + 12 * (x0^2 + x1^2) + 4]
    def second_order_differential_in_dot(self, x):
        return np.array([[12 * x[0]**2 + 4, -8 * x[0] * x[1]],
                         [-8 * x[0] * x[1], 12 * x[1]**2 + 4]])


