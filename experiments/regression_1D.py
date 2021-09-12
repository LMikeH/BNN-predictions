import matplotlib.pyplot as plt
from preds.glm_wrappers.mlp import GLMRegressor
import numpy as np


noise = 0.5
ndata = 100


def f(x):
    return np.cos(x/12)*np.sin(x/9) + 2*np.exp(-np.abs(x-20)/3) 


def data_gen(n):
    xdata = np.sort(100*np.random.rand(n))
    ydata = f(xdata) + np.random.normal(0, noise, size=xdata.shape)

    return xdata.reshape(-1, 1), ydata.reshape(-1, 1)


x, y = data_gen(ndata)


layers = [1, 50, 40, 25, 1]
bnn = GLMRegressor(layers, noise)
bnn.fit(x, y)
bnn.infer()

X = np.linspace(0, 100, 1000)
Y = f(X)


mu, sigma = bnn.predict(X.reshape(-1, 1))

plt.plot(x, y, 'o')
plt.plot(X, Y, '--k')
plt.plot(X, mu)
plt.fill_between(X, mu.flatten() - sigma.flatten(), mu.flatten() + sigma.flatten(), alpha=0.25)
plt.savefig('1dbnn.png', dpi=500)