import numpy as np

def grad_finite_diff(function, w, eps=1e-8):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    result = np.zeros(w.shape[0])
    for i in range(w.shape[0]):
        e_i = np.zeros(w.shape[0])
        e_i[i] += 1
        result[i] = (function(w + e_i * eps) - function(w)) / eps
    return result