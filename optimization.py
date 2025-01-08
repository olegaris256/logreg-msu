import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.special import expit
import oracles


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций

        **kwargs - аргументы, необходимые для инициализации
        """
        if loss_function != 'binary_logistic':
            raise ValueError()
        self.loss_function = loss_function
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.oracle = oracles.BinaryLogistic(**kwargs)
        self.w = None

    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        if w_0 is not None:
            self.w = w_0
        else:
            self.w = np.zeros(X.shape[1])

        history = {
            'time': [0],
            'func': [self.get_objective(X, y)],
            'accuracy': [(self.predict(X) == y).sum() / len(y)]
        }

        for k in range(1, self.max_iter + 1):
            start_time = time.time()
            grad = self.get_gradient(X, y)
            step = self.step_alpha / (k ** self.step_beta)
            self.w -= step * grad

            history['time'].append(time.time() - start_time)
            history['func'].append(self.get_objective(X, y))
            history['accuracy'].append((self.predict(X) == y).sum() / len(y))

            if abs(history['func'][-1] - history['func'][-2]) < self.tolerance:
                break

        if trace:
            return history

    def predict(self, X):
        """
        Получение меток ответов на выборке X

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: одномерный numpy array с предсказаниями
        """
        return np.sign(X @ self.w)

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        proba = expit(X @ self.w)
        return np.vstack((1 - proba, proba)).T

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: float
        """
        return self.oracle.func(X, y, self.w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: numpy array, размерность зависит от задачи
        """
        return self.oracle.grad(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, batch_size, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        batch_size - размер подвыборки, по которой считается градиент

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций (эпох)

        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.

        **kwargs - аргументы, необходимые для инициализации
        """
        super().__init__(loss_function, step_alpha, step_beta, tolerance, max_iter, **kwargs)
        self.batch_size = batch_size
        self.random_seed = random_seed

    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}

        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.

        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        np.random.seed(self.random_seed)
        if w_0 is not None:
            self.w = w_0
        else:
            self.w = np.zeros(X.shape[1])

        history = {
            'epoch_num': [0],
            'time': [0],
            'func': [self.get_objective(X, y)],
            'accuracy': [(self.predict(X) == y).sum() / len(y)],
            'weights_diff': [0]
        }

        processed_current = 0
        n_samples = X.shape[0]
        for k in range(1, self.max_iter + 1):
            shuffled_indices = np.random.permutation(n_samples)
            for i in range(0, n_samples, self.batch_size):
                if i + self.batch_size < n_samples:
                    processed_current += self.batch_size
                else:
                    processed_current += n_samples - i
                curr_epoch_num = processed_current / n_samples

                start_time = time.time()

                current_indices = shuffled_indices[i:i + self.batch_size]
                X_batch = X[current_indices]
                y_batch = y[current_indices]

                grad = self.get_gradient(X_batch, y_batch)
                step = self.step_alpha / (curr_epoch_num ** self.step_beta)
                prev_w = self.w
                self.w -= step * grad

                if curr_epoch_num - history['epoch_num'][-1] > log_freq:
                    history['epoch_num'].append(curr_epoch_num)
                    history['time'].append(time.time() - start_time)
                    history['func'].append(self.get_objective(X, y))
                    history['accuracy'].append((self.predict(X) == y).sum() / len(y))
                    history['weights_diff'].append(np.linalg.norm(prev_w - self.w) ** 2)

                    if abs(history['func'][-1] - history['func'][-2]) < self.tolerance:
                        break
            if len(history['func']) > 1 and abs(history['func'][-1] - history['func'][-2]) < self.tolerance:
                break
        if trace:
            return history
