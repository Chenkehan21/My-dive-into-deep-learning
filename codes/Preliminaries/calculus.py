from os import O_NDELAY
import numpy as np
import matplotlib.pyplot as plt
import math

# Differentiation, the process of finding a derivative in mathematics
def diff(f, x):
    h = 1e-5
    return (f(x + h) - f(x - h)) / (2 * h)

def set_axes(axes, xlabel, ylabel, 
             xlim, ylim, 
             xscale, yscale, 
             xticks, yticks,
             legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if hasattr(xticks, "__len__"):
        axes.set_xticks(xticks)
    if hasattr(yticks, "__len__"):
        axes.set_yticks(yticks)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(x, y=None,
        xlabel=None, ylabel=None, 
        xlim=None, ylim=None, 
        xscale=None, yscale=None,
        xticks=None, yticks=None,
        axes=None, legend=None,
        formats=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), name=None):
    def has_one_axis(x):
        return (hasattr(x, 'ndim') and x.ndim == 1 or 
        isinstance(x, list) and not hasattr(x[0], '__len__'))
    
    axes = axes if axes else plt.gca()
    
    if legend is None:
        legend = []
    
    if has_one_axis(x):
        x = [x]
    if y is None:
        x, y = [[]] * len(x), x
    elif has_one_axis(y):
        y = [y]
    if len(x) != len(y):
        x = x * len(y)
    
    axes.cla()
    for m, n, f in zip(x, y, formats):
        if len(m):
            axes.plot(m, n, f)
        else:
            axes.plot(n, f)
    set_axes(axes, 
             xlabel, ylabel, 
             xlim, ylim, 
             xscale, yscale, 
             xticks, yticks,
             legend)
    if name == None:
        plt.savefig('./calculus.png')
    else:
        plt.savefig('./%s.png'%name)


def get_line(k, x, y):
    if k == 0:
        f = lambda x: y
        return f
    elif math.isinf(k):
        exit("tangent is infinity")
    else:
        f = lambda m: k * m + (y - k * x)
        return f


def exercise1():
    x = np.arange(-2, 2, 0.1)
    f1 = lambda x: x**3 - 1/x
    
    X = 1
    Y = f1(X)
    k = diff(f1, X)

    f2 = get_line(k, X, Y)

    params = {
        'x': x,
        'y': [f1(x), f2(x)],
        'xlabel': 'x',
        'ylabel': 'f(x)', 
        'xlim': (0., 2.), 
        'ylim': (-10., 10.), 
        'xscale': 'linear', 
        'yscale': 'linear',
        # 'xticks': np.linspace(0, 2, 10),
        # 'yticks': np.linspace(-10., 10., 15),
        'axes': plt.gca(), 
        'legend': ['f(x)', 'Tangent line(x=1)'],
        'name': 'exercise1'
    }
    plot(**params)


if __name__ == "__main__":
    exercise1()