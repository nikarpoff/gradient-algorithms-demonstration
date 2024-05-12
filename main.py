import numpy as np

# graphics
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# target functions
import targetfunctions as tfs

# optimizers
import optimizers as opts

# drawers
import drawers


class Main:
    """
    Управляющий класс. Выполняет задачу создания целевых функций, их оптимизаторов, рисовщиков графиков.

    Для использования необходимо инициализировать класс с параметрами initialX - начальное приближение 
    и необязательными startX=-10.0 (правая граница отрисовки графика), endX=10.0 (левая граница отрисовки графика), 
    step=0.1 (шаг построения графика).

    Затем необходимо инициализировать необходимую функцию с помощью методов типа init<Required Function>.
    Может быть выбрана лишь одна функция.

    Далее следует добавить оптимизаторы с помощью методов типа add<Required Optimizer>, которых может быть сколь угодно много.
    После этого можно выполнить функцию drawSolutions() для отрисовки решения оптимизаторов. Все оптимизаторы начинают в одной точке.

    Отрисовка решений возможна только для трехмерных функций. В противном случае возникнет исключение.
    """

    def __init__(self, target_function, initial_x, start_x=-10.0, end_x=10.0, step=0.1):
        # генерируем трехмерную систему координат
        self.fig = plt.figure()
        self.ax_3d = self.fig.add_subplot(projection='3d')
        self.optimizers = []
        self.optimizers_drawers = []

        self.initial_x = initial_x
        self.start_x = start_x
        self.end_x = end_x
        self.step = step

        self.target_function = target_function
        drawers.GraphicDrawer(self.target_function, self.fig, self.ax_3d, self.start_x, self.end_x, self.step).draw()

    def add_gd_optimizer(self, linestyle="ro-", alpha=None):
        """
        Вызовите этот метод, чтобы добавить оптимизатор с использованием наискорейшего спуска.
        Укажите тип линии для рисовщика решения, например по умолчанию установлена красная сплошная линия: "ro-"
        """
        gd_optimizer = opts.GDOptimizer(self.target_function, initial_x=self.initial_x, alpha=alpha)
        gd_optimizer_drawer = drawers.OptimizerDrawer(gd_optimizer, self.ax_3d, linestyle=linestyle)

        self.optimizers.append(gd_optimizer)
        self.optimizers_drawers.append(gd_optimizer_drawer)

    def add_cgd_optimizer(self, linestyle="bo-"):
        """
        Вызовите этот метод, чтобы добавить оптимизатор с использованием метода сопряженных градиентов.
        Укажите тип линии для рисовщика решения, например по умолчанию установлена синяя сплошная линия: "bo-"
        """
        cgd_optimizer = opts.CGDOptimizer(self.target_function, initial_x=self.initial_x)
        cgd_optimizer_drawer = drawers.OptimizerDrawer(cgd_optimizer, self.ax_3d, linestyle=linestyle)

        self.optimizers.append(cgd_optimizer)
        self.optimizers_drawers.append(cgd_optimizer_drawer)

    def add_adam_optimizer(self, linestyle="go-", alpha=0.05, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Вызовите этот метод, чтобы добавить оптимизатор с использованием метода ADAM.
        Укажите параметры betta1, betta2 и epsilon (по-умолчанию - beta1=0.9, beta2=0.999, epsilon=1e-8).
        О предназначении параметров см. optimizers.AdamOptimizer
        Укажите тип линии для рисовщика решения, например по умолчанию установлена зеленая сплошная линия: "go-"
        """
        adam_optimizer = opts.AdamOptimizer(self.target_function, initial_x=self.initial_x, alpha=alpha, beta1=beta1,
                                            beta2=beta2, epsilon=epsilon)

        adam_optimizer_drawer = drawers.OptimizerDrawer(adam_optimizer, self.ax_3d, linestyle=linestyle)

        self.optimizers.append(adam_optimizer)
        self.optimizers_drawers.append(adam_optimizer_drawer)

    def add_momentum_optimizer(self, linestyle="bo-", alpha=0.05, betta=0.9):
        """
        Вызовите этот метод, чтобы добавить оптимизатор с использованием метода импульсов.
        Укажите параметр betta.
        О предназначении параметров см. optimizers.MomentumOptimizer
        Укажите тип линии для рисовщика решения, например по умолчанию установлена синяя сплошная линия: "bo-"
        """
        momentum_optimizer = opts.MomentumOptimizer(self.target_function, initial_x=self.initial_x, momentum_rate=betta, alpha=alpha)

        momentum_optimizer_drawer = drawers.OptimizerDrawer(momentum_optimizer, self.ax_3d, linestyle=linestyle)

        self.optimizers.append(momentum_optimizer)
        self.optimizers_drawers.append(momentum_optimizer_drawer)

    def add_sqn_optimizer(self, linestyle="ro-"):
        """
        Вызовите этот метод, чтобы добавить оптимизатор с использованием алгоритма SQN.
        Укажите тип линии для рисовщика решения, например по умолчанию установлена красная сплошная линия: "ro-"
        """
        newton_optimizer = opts.SQNOptimizer(self.target_function, initial_x=self.initial_x)
        newton_optimizer_drawer = drawers.OptimizerDrawer(newton_optimizer, self.ax_3d, linestyle=linestyle)

        self.optimizers.append(newton_optimizer)
        self.optimizers_drawers.append(newton_optimizer_drawer)

    def add_bfgs_optimizer(self, linestyle="mo-"):
        """
        Вызовите этот метод, чтобы добавить оптимизатор с использованием алгоритма BFGS.
        Укажите тип линии для рисовщика решения, например по умолчанию установлена фиолетовая сплошная линия: "mo-"
        """
        newton_optimizer = opts.BFGSOptimizer(self.target_function, initial_x=self.initial_x)
        newton_optimizer_drawer = drawers.OptimizerDrawer(newton_optimizer, self.ax_3d, linestyle=linestyle)

        self.optimizers.append(newton_optimizer)
        self.optimizers_drawers.append(newton_optimizer_drawer)

    def find_solutions(self):
        for i in range(len(self.optimizers)):
            self.optimizers[i].find_solution()

    def draw_solutions(self):
        anis = []

        for i in range(len(self.optimizers_drawers)):
            anis.append(
                FuncAnimation(self.fig, self.optimizers_drawers[i].draw_solution, frames=np.arange(0, 100), blit=True,
                              repeat=False))

        plt.tight_layout()
        plt.show()


def init_paraboloid(A, b, c):
    """
    Вызовите этот метод, чтобы инициализировать эллиптический параболоид.
    """
    return tfs.EllipticalParaboloid(A, b, c)


def init_oscillator():
    """
    Вызовите этот метод, чтобы инициализировать двумерную осциллирующую функцию.
    """
    return tfs.OscillatoryBivariateFunction()


def init_unimod():
    """
    Вызовите этот метод, чтобы инициализировать унимодальную функцию с ямкой (для демонстрации методов с импульсами).
    """
    return tfs.UnimodalPitFunction()


if __name__ == "__main__":
    # example with paraboloid
    # main = Main(init_paraboloid(A=[[3, 0], [0, 1]], b=[0, 0], c=5), [-7.5, 12])

    # example with oscillator
    main = Main(init_oscillator(), [1.9, 0.1], -1.0, 5.0)

    # example with unimod
    # main = Main(init_unimod(), [0.5, 0.10], -0.5, 1.5)

    main.add_gd_optimizer(alpha=0.2)
    main.add_cgd_optimizer()
    main.add_momentum_optimizer(alpha=0.2)
    main.add_adam_optimizer(alpha=0.5, beta1=0.95, beta2=0.999)
    main.add_sqn_optimizer()
    main.add_bfgs_optimizer()

    # testing solutions:
    main.find_solutions()

    # main.draw_solutions()
