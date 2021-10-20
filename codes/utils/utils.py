import torch
import matplotlib.pyplot as plt
import random


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
        plt.savefig('./tmp.png')
    else:
        plt.savefig('./%s.png'%name)


def data_iter(batch_size, features, labels):
    num_samples = len(features)
    indices = list(range(num_samples))
    random.shuffle(indices)
    for i in range(0, num_samples, batch_size):
        index_mask = indices[i : min(i + batch_size, num_samples)]
        yield features[index_mask], labels[index_mask]
