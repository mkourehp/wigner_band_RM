import matplotlib.pyplot as plt
import numpy as np
import functools


def print_h(h: np.array):
    for row in h:
        [print("{:10.3}".format(r), end="\t") for r in row] ; print()


def plot_h(h):
    plt.imshow(h)
    plt.colorbar()

def normalize(p): 
    return p / np.sqrt(np.sum(np.abs(p)**2))


def fit_line(x: np.array, y: np.array, plot: bool=False, dE: float=False) -> dict:
    x, y = x[1:], y[1:]
    def _get_range_index()->int:
        dy = normalize(np.diff(y, 1))
        for i in range(dy.size):
            if dy[i] <= 0. and i > 1: return i + 1
        return dy.size
    index = _get_range_index()
    p, res, _, _, _ = np.polyfit(x=x[:index], y=y[:index], deg=1, full=True)
    if not plot:
        return {"slope":p[0],"residual":res[0] if res else 0}
    plt.plot(x, y, "-", zorder=10,label=f"dE:{dE or 0:.0f}, slope:{p[0]:.2f}")
    # plt.axvline(ymin=0, ymax=0.5, x=x[index], alpha=0.5, c="C10")
    plt.plot(x[:index], x[:index]*p[0]+p[1], "--", c="grey", alpha=0.5, zorder=-10)
    plt.ylabel("S(t)",  size=15)
    plt.xlabel("t",  size=15)
    plt.legend()
    return {"slope":p[0],"residual":res[0] if res else 0}


def ensemble_iterator(iterate: int):
    def ensemble_decorator_iterator(func):
        @functools.wraps(func)
        def ensemble_wrapper_iterator(*args, **kwargs):
            for _ in range(iterate):
                value = func(*args, **kwargs)
            return value
        return ensemble_wrapper_iterator
    return ensemble_decorator_iterator