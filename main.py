import argparse
import copy
import json

import numpy as np
import scipy
import time
from sklearn.datasets import load_svmlight_file


default_reg = 1e-4

def get_dataset(ds_path):
    if "a1a" in ds_path:
        data = load_svmlight_file(ds_path)
        y = np.array(data[1] >= 0, dtype=np.int32).reshape(-1, 1)
        return data[0], y
    elif "breast-cancer_scale" in ds_path:
        data = load_svmlight_file(ds_path)
        y = np.array(data[1] == 4, dtype=np.int32).reshape(-1, 1)
        return data[0], y
    else:
        np.random.seed(42)
        a, b = np.random.random() * 2 - 1, np.random.random() * 2 - 1
        X = np.random.normal(size=1000).reshape(-1, 1)
        y = np.array((a * X + b) >= 0, dtype=np.int32)
        X = scipy.sparse.csr_matrix(X)
        return X, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def add_ones(X):
    return scipy.sparse.hstack((X, np.ones(X.shape[0]).reshape(-1, 1)))


def calls_count_cache(oracle):
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return oracle(*args, **kwargs)
    wrapper.calls = 0
    return wrapper


def oracle_0_for_minimize(X, y, w):
    if w.shape != (X.shape[1], 1):
        w = np.array(w).reshape(-1, 1)
    n = y.shape[0]
    z = X @ w
    return (-(y.T @ z - np.sum(np.logaddexp(0, z))) / n + 0.1 * w.T @ w)[0, 0]


@calls_count_cache
def oracle_0(X, y, w, reg=default_reg):
    n = y.shape[0]
    z = X @ w
    return (-(y.T @ z - np.sum(np.logaddexp(0, z))) / n + reg * w.T @ w)[0, 0]


@calls_count_cache
def oracle_1(X, y, w, reg=default_reg):
    n = y.shape[0]
    return X.T @ (sigmoid(X @ w) - y) / n + 2 * reg * w


@calls_count_cache
def oracle_2(X, y, w, reg=default_reg):
    n = y.shape[0]
    z = sigmoid(X @ w)
    return X.T @ np.multiply(np.eye(X.shape[0]), np.multiply(z, (1 - z))) @ X / n + 2 * reg * np.eye(len(w))


# line search methods


def golden_search(f, left, right, eps):
    phi = (1 + 5 ** 0.5) / 2
    resphi = 2 - phi
    x1 = left + resphi * (right - left)
    x2 = right - resphi * (right - left)
    f1 = f(x1)
    f2 = f(x2)
    while right - left > eps:
        if f1 < f2:
            right = x2
            x2 = x1
            f2 = f1
            x1 = left + resphi * (right - left)
            f1 = f(x1)
        else:
            left = x1
            x1 = x2
            f1 = f2
            x2 = right - resphi * (right - left)
            f2 = f(x2)
    xmin = (x1 + x2) / 2
    return xmin


def grad_ratio(grad, init_grad):
    grad = np.abs(grad)
    grad = grad.T @ grad
    init_grad = np.abs(init_grad)
    init_grad = init_grad.T @ init_grad
    return (grad / init_grad)[0, 0]


def step_oracle_0(X, y, w, oracle_0, direction):
    return lambda step: oracle_0(X, y, w - step * direction)


def step_oracle_1(X, y, w, oracle_1, direction):
    return lambda step: (-direction.T @ oracle_1(X, y, w - step * direction))[0, 0]


# line-search


def golden(X, y, w, oracles, direction, guess=10, init=10, eps=1e-6):
    return golden_search(step_oracle_0(X, y, w, oracles[0], direction), 0, init, eps)


def brent(X, y, w, oracles, direction, guess=10, init=10, eps=1e-6):
    bounds = (0, init)
    res = scipy.optimize.minimize_scalar(step_oracle_0(X, y, w, oracles[0], direction),
                                         bracket=bounds, bounds=bounds, method='Brent', tol=eps)
    return res.x


def armijo(X, y, w, oracles, direction, guess=10, init=10, c=0.01):
    step_func = step_oracle_0(X, y, w, oracles[0], direction)
    zero_value = oracles[0](X, y, w)
    zero_grad = step_oracle_1(X, y, w, oracles[1], direction)(0)
    while step_func(guess) > zero_value + c * guess * zero_grad:
        guess /= 2
    return guess


def wolfe(X, y, w, oracles, direction, guess=10, init=10, c1=0.001, c2=0.9):
    res = scipy.optimize.linesearch.scalar_search_wolfe2(step_oracle_0(X, y, w, oracles[0], direction),
                                                         step_oracle_1(X, y, w, oracles[1], direction),
                                                         c1=c1, c2=c2)[0]
    return res


def lipschitz(X, y, w, oracles, direction, guess=10, init=10):
    step_func = step_oracle_0(X, y, w, oracles[0], direction)
    zero_value = step_func(0)
    zero_grad = step_oracle_1(X, y, w, oracles[1], direction)(0)
    while step_func(guess) > zero_value + 0.5 * guess * zero_grad:
        guess /= 2
    return guess


# methods


def sgd(X, y, w, oracles, **params):
    return oracles[1](X, y, w)


def newton(X, y, w, oracles, **params):
    cho_fac = scipy.linalg.cho_factor(oracles[2](X, y, w))
    return scipy.linalg.cho_solve(cho_fac, oracles[1](X, y, w))


def hessian_times_vector(X, y, w, vector, reg=0.1):
    n = y.shape[0]
    z = sigmoid(X @ w)
    B = np.multiply(np.eye(n), np.multiply(z, (1 - z)))
    res = X.T @ (B @ (X @ vector)) / n + 2 * reg * vector
    return res


def newton_free(X, y, w, oracles, policy='const', eta=0.5, iters=100):
    grad = oracles[1](X, y, w)
    grad_norm = np.linalg.norm(grad)
    tol = grad_norm * eta
    if policy == 'sqrtGradNorm':
        tol = tol * min(grad_norm ** 0.5, 0.5)
    elif policy == 'gradNorm':
        tol = tol * min(grad_norm, 0.5)
    z = np.zeros_like(w)
    g = hessian_times_vector(X, y, w, z) + grad
    d = -g
    Hd = hessian_times_vector(X, y, w, d)
    for i in range(iters):
        gamma = (g.T @ g / (d.T @ Hd))[0, 0]
        z = z + gamma * d
        g_n = g + gamma * Hd
        if np.linalg.norm(g_n) < tol:
            return -z
        beta = ((g_n.T @ g_n) / (g.T @ g))[0, 0]
        d = -g + beta * d
        g = g_n
        Hd = hessian_times_vector(X, y, w, d)
    return z


def bfgs(X, y, w, oracles, prev_w=None, prev_grad=None, prev_B=None):
    grad = oracles[1](X, y, w)
    if prev_B is None:
        prev_B = np.eye(len(w))
        prev_w = np.zeros_like(w)
        prev_grad = np.zeros_like(grad)
    s = w - prev_w
    y = grad - prev_grad
    first = (s.T @ y + y.T @ prev_B @ y) * (s @ s.T) / ((s.T @ y)[0, 0] ** 2)
    second = (prev_B @ y @ s.T + s @ y.T @ prev_B) / ((s.T @ y)[0, 0])
    B = prev_B + first - second
    params = {'prev_B': B, 'prev_w': w, 'prev_grad': grad}
    direction = B @ grad
    return direction, params


def lbfgs(X, y, w, oracles, history_size=5, ss=None, ys=None, prev_w=None, prev_grad=None):
    grad = oracles[1](X, y, w)
    if ss is None:
        prev_w = np.array(w)
        prev_grad = np.array(grad)
        ss = [prev_w] * history_size
        ys = [prev_grad] * history_size
    else:
        ss = ss[1:] + [w - prev_w]
        ys = ys[1:] + [grad - prev_grad]

    q = grad
    aa = [0] * history_size

    for i in range(history_size):
        s = ss[history_size - i - 1]
        y = ys[history_size - i - 1]
        a = s.T @ q / (y.T @ s)
        aa[history_size - i - 1] = a
        q = q - a * y

    gamma = ss[0].T @ ys[0] / (ys[0].T @ ys[0])
    H = gamma * np.eye(len(w))
    z = H @ q
    for i in range(history_size):
        s = ss[i]
        y = ys[i]
        a = aa[i]
        b = y.T @ z / (y.T @ s)
        z = z + s * (a - b)

    params = {'history_size': history_size, 'ss': ss, 'ys': ys, 'prev_w': w, 'prev_grad': grad}
    return z, params


# optimization


def optimize(X, y, w, oracles, compute_direction, ls_method, max_iters=100, eps=1e-5):
    get_direction, dir_params = compute_direction
    compute_step, ls_params = ls_method
    init_grad = oracles[1](X, y, w)
    opt_arg = scipy.optimize.minimize(lambda x: oracle_0_for_minimize(X, y, x), w).x.reshape(-1, 1)
    opt_value = oracles[0](X, y, opt_arg)
    r1 = [abs(opt_value - oracles[0](X, y, w))]
    r2 = [1]
    times = [0]
    # calls = [0]
    start_time = time.process_time()
    step = 50
    not_converge_at_least_once = False
    for oracle in oracles:
        oracle.calls = 0
    for iter_num in range(max_iters):
        if r2[-1] < eps:
            iter_num -= 1
            break
        direction, dir_params = get_direction(X, y, w, oracles, **dir_params)
        step = compute_step(X, y, w, oracles, direction, 2 * step, 100, *ls_params)
        if step is None:
            if not not_converge_at_least_once:
                print("Line search is not converges")
            not_converge_at_least_once = True
            step = 0.
        if oracles[0](X, y, w) is None or np.isnan(oracles[0](X, y, w)):
            w = opt_arg
            # print(iter_num, direction)
            break
        w = w - step * direction
        r1.append(abs(opt_value - oracles[0](X, y, w)))
        r2.append(grad_ratio(oracles[1](X, y, w), init_grad))
        times.append(time.process_time() - start_time)
        # calls.append(sum(oracle.calls for oracle in oracles))
    calls = {i: oracles[i].calls for i in range(3)}
    return w, r1, r2, times, list(range(iter_num + 2)), calls


class LogisticRegression:
    def __init__(self, init_distribution, opt_method, ls_method, cg_tolerance_eta, cg_tolerance_policy,
                 lbfgs_history_size, eps=1e-5):
        self.oracles = [oracle_0, oracle_1, oracle_2]
        self.ls_methods = {'golden_search': golden, 'brent': brent, 'armijo': armijo, 'wolfe': wolfe,
                           'lipschitz': lipschitz}
        self.ls_methods_args = {'golden_search': tuple(), 'brent': tuple(), 'armijo': (0.05,), 'wolfe': (0.01, 0.9),
                           'lipschitz': tuple()}
        self.opt_methods = {'gradient': sgd, 'newton': newton, 'hfn': newton_free, 'bfgs': bfgs, 'lbfgs': lbfgs}
        self.opt_methods_args = {'gradient': {}, 'newton': {},
                                 'hfn': {'iters': 30, 'policy': cg_tolerance_policy, 'eta': cg_tolerance_eta},
                                 'bfgs': {}, 'lbfgs': {'history_size': lbfgs_history_size}}
        self.init_distribution = init_distribution
        self.opt_method = opt_method
        self.ls_method = ls_method
        self.max_iters = 100
        self.eps = eps
        self.X = None
        self.W = None
        self.opt_res = {}

    def init_weights(self, X):
        if self.W is not None:
            return
        if self.X is None:
            return
        if self.init_distribution == "uniform":
            self.W = (np.random.rand(X.shape[1]) * 2 - 1).reshape(-1, 1)
        elif self.init_distribution == "gaussian":
            self.W = np.random.normal(0, 10 ** 0.5, X.shape[1]).reshape(-1, 1)
        else:
            raise ValueError("Wrong value of point_distribution option")

    def fit(self, X, y):
        self.X = add_ones(X)
        self.y = y
        if self.W is None:
            self.init_weights(self.X)
        self.opt_res['initial_point'] = list(copy.deepcopy(self.W).reshape(-1))
        res = optimize(self.X, self.y, self.W, self.oracles,
                 (self.opt_methods[self.opt_method], self.opt_methods_args[self.opt_method]),
                 (self.ls_methods[self.ls_method], self.ls_methods_args[self.ls_method]),
                  max_iters=self.max_iters, eps=self.eps)
        w, r1, r2, times, iter_num, calls = res
        self.W = w
        self.opt_res['optimal_point'] = list(w.reshape(-1))
        self.opt_res['func_value'] = oracle_0(self.X, y, w)
        self.opt_res['gradient_value'] = np.linalg.norm(oracle_1(self.X, y, w).reshape(-1))
        ind2name = {0: 'f', 1: 'df', 2: 'd2f'}
        self.opt_res['oracle_calls'] = {ind2name[i]: calls[i] for i in range(3)}
        self.opt_res['r_k'] = r2[-1]
        self.opt_res['working_time'] = times[-1]
        return self.opt_res

    def predict(self, X):
        if self.W is None:
            self.init_weights(X)
        return add_ones(X) @ self.W


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_path", help="path to dataset file in .svm format")
    parser.add_argument("--optimize_method", help="high-level optimization method, will be one of"
                                                  " {'gradient', 'newton', 'hfn'}")
    parser.add_argument("--line_search", help="linear optimization method, will be one of "
                                              "{'golden_search', 'brent', 'armijo', 'wolfe', 'lipschitz'}")
    parser.add_argument("--point_distribution", default='uniform', help="initial weights distribution class, "
                                                                        "will be one of {'uniform', 'gaussian'}")
    parser.add_argument("--seed", type=int, default=3, help="seed for numpy randomness")
    parser.add_argument("--eps", type=float, default=1e-6, help="epsilon to use in termination condition")
    parser.add_argument("--cg_tolerance_eta", type=float, default=0.5, help="epsilon to use in termination condition")
    parser.add_argument("--cg_tolerance_policy", default='const')
    parser.add_argument("--lbfgs_history_size", default=5, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    X, y = get_dataset(args.ds_path)
    model = LogisticRegression(args.point_distribution,
                               args.optimize_method,
                               args.line_search,
                               args.cg_tolerance_eta,
                               args.cg_tolerance_policy,
                               args.lbfgs_history_size,
                               args.eps)
    opt_res = model.fit(X, y)
    opt_res = json.dumps(opt_res, indent=4)
    f = open("out.json", 'w')
    f.write(opt_res)
    f.close()


# --ds_path "data/a1a.txt" --line_search golden_search --point_distribution uniform
# --ds_path "../data/a1a.txt" --line_search brent --point_distribution gaussian --optimize_method hfn
if __name__ == "__main__":
    main()
