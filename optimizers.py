import numpy as np
import scipy as sp
import time
import random

from targetfunctions import TargetFunction
from util import count_a_step
from util import transposeVector


class Optimizer:
    """
    Данный класс представляет обобщение для всех оптимизаторов.

    Класс предназначен для нахождения минимума целевой функции из точки initialX.

    Поиск может осуществляться как целиком за раз (функция findSolution), так и по шагам
    (функция makeStep), что используется в рисовщиках алгоритмов минимизации.
    """

    def __init__(self, func: TargetFunction, initial_x):
        self.x = None
        self.target_function = func
        self.gradient = func.differential_in_dot(initial_x)

        self.initial_x = initial_x

    def find_solution(self, accuracy=0.05):
        """
        Функция предназначена для поиска минимума целевой функции (TargetFunction) из точки initialX,
        а также вывода количества шагов и потребовавшегося времени. Возвращает вектор точек [x, y]
        """

        print("=" * 16, "\n")

        print(f"Running {self.__class__.__name__}. Accuracy: {accuracy}")
        self.additional_info()

        step = 0
        y = 0.

        not_satisfied_accuracy = True
        start_time = time.time()

        while not_satisfied_accuracy:
            y = self.make_step()[-1]
            current_accuracy = np.linalg.norm(self.gradient)

            if current_accuracy < accuracy:
                not_satisfied_accuracy = False

            step += 1

        end_time = time.time()

        print(f"\tTotal steps: {step}")
        print(f"\tSolution: {y}")
        print(f"\tRequested time: {end_time - start_time}")

        print()
        print("=" * 16, "\n")

        return [self.x, y]

    def make_step(self):
        """
        Функция предназначена для выполнения одного шага поиска минимума целевой функции
        (TargetFunction) из точки initialX, с учетом предыдущих шагов
        """
        raise NotImplementedError("make_step method must be implemented in child classes")

    def additional_info(self):
        """
        Функция вывода дополнительной информации об оптимизаторе, например использование динамического шага,
        специфические константы и др.
        """
        pass


class GDOptimizer(Optimizer):
    """
    Данный класс представляет простейший оптимизатор с использованием наискорешего спуска (Gradient Descent Optimizer).

    Если в конструкторе задан статический шаг, то будет использоваться он. В противном случае будет высчитываться
    динамический шаг по методу Ньютона-Рафсона (метод второго порядка в данном случае)

    На каждом шаге подсчитывается антиградиент в точке и динамический шаг, а затем с данным шагом
    осуществляется переход по вектору x в направлении антиградиента.
    """

    def __init__(self, func: TargetFunction, initial_x, alpha):
        super().__init__(func, initial_x)
        self.x = self.initial_x
        self.gradient = self.target_function.differential_in_dot(self.x)

        # Булева переменная об использовании динамического шага
        self.is_dynamic_step_used = False

        if alpha is None:
            self.is_dynamic_step_used = True
        else:
            self.alpha = alpha

    def make_step(self):
        # подсчитываем динамический шаг, если не указано, что его нужно считать статическим, по методу Ньютона-Рафсона
        if self.is_dynamic_step_used:
            self.alpha = count_a_step(self.gradient, self.target_function.second_order_differential_in_dot(self.x))

        self.x = self.x - self.alpha * self.gradient
        y = self.target_function(self.x)

        self.gradient = self.target_function.differential_in_dot(self.x)

        return [self.x, y]


class CGDOptimizer(Optimizer):
    """
    Данный класс представляет оптимизатор с использованием метода сопряженых градиентов (Conjugate Gradient Descent Optimizer).

    На каждом шаге подсчитывается антиградиент в точке и динамический шаг, а затем с данным шагом
    осуществляется переход по вектору x в направлении антиградиента, далее высчитывается градиент
    в новой точке, высчитывается коэффициент betta, учитывающий новый и старый градиенты, с его
    использованием уточняется новый антиградиент (newAntiGradient = - newGradient + betta * oldAntiGradient).
    """

    def __init__(self, func: TargetFunction, initial_x):
        super().__init__(func, initial_x)
        # Начальное приближение
        self.x = self.initial_x

        # Номер итерации
        self.n = 0

        # Градиент и антиградиент в точке начального приближения
        self.gradient = self.target_function.differential_in_dot(self.x)

    def make_step(self):
        # Высчитываем динамический шаг (см. util.count_a_step)
        alpha = count_a_step(self.gradient, self.target_function.second_order_differential_in_dot(self.x))

        # Находим новое приближение
        self.x = self.x - alpha * self.gradient
        y = self.target_function(self.x)

        # Находим градиент в точке нового приближения
        new_gradient = self.target_function.differential_in_dot(self.x)

        # Не на нулевой итерации вычисляем сопряженное направление, обновляем градиент
        if self.n != 0:
            # Знаменатель
            det = self.gradient.dot(self.gradient)

            # Для избежания деления на 0
            if det == 0:
                beta = 0
            else:
                # Формула Флетчера-Ривса
                beta = new_gradient.dot(new_gradient) / det

                # Сброс каждые n + 1 шагов
                if self.n % (self.x.shape[0] + 1) == 0:
                    beta = 0

            # Обновляем градиент
            self.gradient = new_gradient + beta * self.gradient
        else:
            self.gradient = new_gradient

        self.n += 1

        return [self.x, y]


class MomentumOptimizer(Optimizer):
    """
    Данный класс представляет оптимизатор с использованием метода моментов

    На каждом шаге подсчитывается антиградиент в точке и динамический шаг, затем вычисляется скользящее среднее,
    содержащее информацию обо всех предыдущих градиентах. Далее вычисляется новое значение x с учетом скользящего
    среднего

    Если в конструкторе задан статический шаг, то будет использоваться он. В противном случае будет высчитываться
    динамический шаг по методу Ньютона-Рафсона (метод второго порядка в данном случае)
    """

    def __init__(self, func: TargetFunction, initial_x, momentum_rate, alpha):
        super().__init__(func, initial_x)

        self.x = self.initial_x

        self.k = 0

        self.gradient = self.target_function.differential_in_dot(self.x)

        # последнее скользящее среднее градиентов (moving_average)
        self.last_ma = 0.
        self.momentum_rate = momentum_rate

        # Булева переменная об использовании динамического шага
        self.is_dynamic_step_used = False

        if alpha is None:
            self.is_dynamic_step_used = True
        else:
            self.alpha = alpha

    def additional_info(self):
        print(f"Momentum rate: {self.momentum_rate}")
        if self.is_dynamic_step_used:
            print("Use dynamic step.")
        else:
            print(f"Use static step: {self.alpha}")

    def make_step(self):
        # подсчитываем динамический шаг, если не указано, что его нужно считать статическим, по методу Ньютона-Рафсона
        if self.is_dynamic_step_used:
            self.alpha = count_a_step(self.gradient, self.target_function.second_order_differential_in_dot(self.x))

        # v = v_n-1 * betta + (1 - betta) * alpha * gradient
        self.last_ma = (self.momentum_rate * self.last_ma + (1 - self.momentum_rate) * self.alpha * self.gradient)

        # x = x - v
        self.x = self.x - self.last_ma

        y = self.target_function(self.x)

        self.gradient = self.target_function.differential_in_dot(self.x)

        return [self.x, y]


class AdamOptimizer(Optimizer):
    """
    Данный класс представляет оптимизатор с использованием метода ADAM (Adaptive Moment Estimation)

    На каждом шаге подсчитывается градиент в точке и динамический шаг (alpha). Затем вычисляются скользящее среднее
    градиентов (с учетом параметра betta1 - коэффициент затухания для скользящего среднего градиентов) и скользящее
    среднее квадратов градиентов (с учетом параметра betta2 - коэффициента затухания для скользящего среднего квадратов
    градиентов). Далее эти переменные нормализуются с учетом номера итерации алгоритма (получаем vn и Gn).
    Далее происходит вычисление обновленного значения для x по формуле x = x - alpha * (vn / (sqrt(Gn) + epsilon)).
    epsilon используется для избежания деления на 0. Далее вычисляется целевая функция (y) и алгоритм повторяется.

    Параметры по умолчанию: betta1=0.9, betta2=0.999, epsilon=1e-8


    Если в конструкторе задан статический шаг, то будет использоваться он. В противном случае будет высчитываться
    динамический шаг по методу Ньютона-Рафсона (метод второго порядка в данном случае)
    """

    def __init__(self, func: TargetFunction, initial_x, alpha, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(func, initial_x)

        self.x = self.initial_x

        self.k = 0

        self.gradient = self.target_function.differential_in_dot(self.x)

        # Булева переменная об использовании динамического шага
        self.is_dynamic_step_used = False

        if alpha is None:
            self.is_dynamic_step_used = True
        else:
            self.alpha = alpha

        # последнее скользящее среднее градиентов (moving_average)
        self.last_ma = 0.
        # последнее скользящее среднее квадратов градиентов (moving_average_of_squares)
        self.last_mas = 0.

        # коэффициент для скользящего среднего
        self.beta1 = beta1

        # коэффициент для скользящего среднего квадратов градиентов
        self.beta2 = beta2

        # слагаемое для избежания деления на 0
        self.epsilon = epsilon

    def additional_info(self):
        print(f"beta1: {self.beta1}, beta2: {self.beta2}.")
        if self.is_dynamic_step_used:
            print("Use dynamic step.")
        else:
            print(f"Use static step: {self.alpha}")

    def make_step(self):
        # подсчитываем динамический шаг, если не указано, что его нужно считать статическим, по методу Ньютона-Рафсона
        if self.is_dynamic_step_used:
            self.alpha = count_a_step(self.gradient, self.target_function.second_order_differential_in_dot(self.x))

        # высчитываем скользящее среднее аналогично методу импульсов
        # v = v_n-1 * betta + (1 - betta) * alpha * gradient
        self.last_ma = self.beta1 * self.last_ma + (1 - self.beta1) * self.gradient

        # высчитываем скользящее среднее квадратов градиентов аналогично методу RMSProp
        # v = v_n-1 * betta + (1 - betta) * alpha * gradient ^ 2
        self.last_mas = self.beta2 * self.last_mas + (1 - self.beta2) * (self.gradient ** 2)

        # Выполняем нормировку скользящих средних (v = v / (1 - betta^(k+1))). С ростом числа проделанных шагов
        # значение нормализованных величин будет уменьшаться.
        normalized_ma = self.last_ma / (1 - np.power(self.beta1, self.k + 1))
        normalized_mas = self.last_mas / (1 - np.power(self.beta2, self.k + 1))

        # x = x - alpha * (v / (sqrt(G) + eps))
        self.x = self.x - self.alpha * (normalized_ma / (np.sqrt(normalized_mas) + self.epsilon))

        y = self.target_function(self.x)

        self.gradient = self.target_function.differential_in_dot(self.x)

        # увеличиваем счетчик шагов
        self.k += 1

        return [self.x, y]


class SQNOptimizer(Optimizer):
    """
    Данный класс представляет оптимизатор SQN.

    На каждом шаге подсчитывается антиградиент в точке, гессиан в точке, динамический шаг, а затем с данным шагом
    осуществляется переход по вектору x в направлении антиградиента, умноженном на гессиан.
    """

    def __init__(self, func: TargetFunction, initial_x):
        super().__init__(func, initial_x)
        self.x = self.initial_x
        self.gradient = self.target_function.differential_in_dot(self.x)

    def make_step(self):
        hessian = self.target_function.second_order_differential_in_dot(self.x)

        # высчитываем направление для движения вдоль антиградиента как матричное умножение якобиана на градиент
        direction = -np.dot(hessian, self.gradient)

        # выполняем линейный поиск константы альфа (шага вдоль направления) для удовлетворения условиям Вольфа
        # фактически, мы ищем такой шаг альфа, при котором целевая функция в точке следующего шага будет минимальна
        line_search = sp.optimize.line_search(self.target_function, self.target_function.differential_in_dot,
                                                self.x, direction)
        alpha = line_search[0]

        if alpha is None:
            alpha = 0.0

        self.x = self.x + alpha * direction
        y = self.target_function(self.x)

        self.gradient = self.target_function.differential_in_dot(self.x)

        return [self.x, y]


class BFGSOptimizer(Optimizer):
    """
    Данный класс представляет оптимизатор с использованием метода SQN (stochastic quasi-Newton), а именно - BFGS.

    На каждом шаге подсчитывается новое направление, динамический шаг, приближение к матрице Гессе.
    """

    def __init__(self, func: TargetFunction, initial_x):
        super().__init__(func, initial_x)

        self.x = self.initial_x

        self.k = 0

        self.gradient = self.target_function.differential_in_dot(self.x)

        # Задаём начальный якобиан в виде единичной матрицы
        self.identity_matrix = np.identity(len(self.x), dtype=float)
        self.H = self.identity_matrix

        # Изменение по аргументам функции
        self.x_delta = np.zeros(len(self.x), dtype=float)

        # Изменение по градиентам
        self.gradient_delta = np.zeros(len(self.x), dtype=float)

    def make_step(self):
        # высчитываем направление для движения вдоль антиградиента как матричное умножение якобиана на градиент
        direction = -np.dot(self.H, self.gradient)

        # запоминаем текущие градиенты и вектор x
        last_x = self.x
        last_gradient = self.gradient

        # выполняем линейный поиск константы альфа (шага вдоль направления) для удовлетворения условиям Вольфа
        # фактически, мы ищем такой шаг альфа, при котором целевая функция в точке следующего шага будет минимальна
        line_search = sp.optimize.line_search(self.target_function, self.target_function.differential_in_dot,
                                              self.x, direction)
        alpha = line_search[0]

        if alpha is None:
            alpha = 0.0

        # движемся вдоль полученного направления с вычисленным шагом
        self.x = self.x + alpha * direction

        # вычисляем значение функции в новой точке и значение градиента в новой точке
        y = self.target_function(self.x)
        self.gradient = self.target_function.differential_in_dot(self.x)

        # обновляем изменение x и градиента (x_delta = x_k+1 - x_k; (g_delta = g_k+1 - g_k))
        self.x_delta = self.x - last_x  # также обозначаем как sk
        self.gradient_delta = self.gradient - last_gradient  # также обозначаем как yk

        # вычисляем константу ро
        ro = 1.0 / np.dot(self.gradient_delta, self.x_delta)

        # -------------------------------------------------------a1------------------------a2
        # выполняем обновление якобиана по формуле: H = (I - ro * ykT * sk) * H * (I - ro * skT * yk) + ro * skT * sk
        a1 = self.identity_matrix - ro * transposeVector(self.gradient_delta) * self.x_delta
        a2 = self.identity_matrix - ro * transposeVector(self.x_delta) * self.gradient_delta

        self.H = np.dot(a1, np.dot(self.H, a2)) + (ro * transposeVector(self.gradient_delta) * self.gradient_delta)

        self.k += 1

        return [self.x, y]
