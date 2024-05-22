import matplotlib.pyplot as plt
import numpy as np


def print_h(h: np.array):
    for row in h:
        [print("{:10.3}".format(r), end="\t") for r in row] ; print()


def plot_h(h):
    plt.imshow(h)
    plt.colorbar()

def normalize(p):
    return p / np.sqrt(np.sum(np.abs(p)**2))


def fit_line(x: np.array, y: np.array, plot: bool=False) -> dict:
    x, y = x[1:], y[1:]
    def _get_range_index()->int:
        dy = normalize(np.diff(y, 1))
        for i in range(dy.size):
            if dy[i] <= 0.: return i
    index = _get_range_index()
    p, res, _, _, _ = np.polyfit(x=x[:index], y=y[:index], deg=1, full=True)
    if not plot:
        return {"slope":p[0],"residual":res[0]}
    plt.clf()
    plt.plot(x[:index], y[:index], "-")
    plt.plot(x[:index], x[:index]*p[0]+p[1], "--", label=f"slope: {p[0]:.2f}, residual: {res[0]:.2f}")
    plt.ylabel("S(t)")
    plt.xlabel("t")
    plt.legend()
    plt.show()
    return {"slope":p[0],"residual":res[0]}
