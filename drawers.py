import numpy as np

from targetfunctions import TargetFunction
from optimizers import Optimizer


class GraphicDrawer:
    """
    Данный класс предназначен для отрисовки целевой функции.

    Но класс не отрисовывает функцию при вызове draw(), а лишь
    добавляет её к уже имеющейся системе координат ax_3d.
    
    Кроме того, класс работает только с трехмерными функциями
    """

    def __init__(self, func: TargetFunction, fig, ax_3d, start_x, end_x, step):
        """
        Укажите целевую функцию типа TargetFunction, fig из библиотеки matplotlib, 
        трехмерную систему координат, в которой будет работать данный класс, а также
        startX (правая граница отрисовки графика), endX (левая граница отрисовки графика), 
        step (шаг построения графика).
        """
        self.targetFunction = func

        self.fig = fig
        self.ax_3d = ax_3d

        self.start_x = start_x
        self.end_x = end_x
        self.step = step

    def draw(self):
        x1 = np.arange(self.start_x, self.end_x, self.step)
        x2 = np.arange(self.start_x, self.end_x, self.step)

        x1grid, x2grid = np.meshgrid(x1, x2)

        data_f = np.empty_like(x1grid)

        for i in range(x1grid.shape[0]):
            for j in range(x1grid.shape[1]):
                data_f[i, j] = self.targetFunction([x1grid[i, j], x2grid[i, j]])

        self.ax_3d.plot_surface(x1grid, x2grid, data_f, cmap='inferno')


class OptimizerDrawer:
    """
    Данный класс предназначен для отрисовки решения оптимизатора по шагам.

    Но класс не отрисовывает решение при вызове drawSolution(), а лишь
    добавляет её к уже имеющейся системе координат ax_3d.
    """

    def __init__(self, optimizer: Optimizer, ax_3d, linestyle):
        """
        Укажите оптимизатор типа Optimizer, трехмерную систему координат, в которой будет работать данный класс, 
        а также стиль линии, которой должен отрисовываться график.
        """
        self.line, = ax_3d.plot([], [], [], linestyle)
        self.points_x = [optimizer.initial_x]
        self.points_y = [optimizer.target_function(optimizer.initial_x)]

        self.optimizer = optimizer

    def draw_solution(self, frame):
        x, y = self.optimizer.make_step()

        self.points_x.append(x)  # Добавляем текущую точку в список
        self.points_y.append(y)  # Добавляем текущую точку в список

        self.line.set_data(*zip(*self.points_x))  # Разделяем координаты точек для отрисовки линий
        self.line.set_3d_properties(self.points_y, 'z')
        return self.line,
