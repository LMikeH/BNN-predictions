import numpy as numpy
import torch
from torch.nn.utils import parameters_to_vector
from torch.optim import Adam

from preds.models import MLPS
from preds.likelihoods import GaussianLh
from preds.datasets import SnelsonGen
from preds.laplace import Laplace


class GLMRegressor(torch.nn.Module):
    def __init__(self, layers, noise):
        super().__init__()
        self.model = MLPS(
            layers[0],
            layers[1:-1],
            layers[-1]
        )
        self.lh = GaussianLh(sigma_noise=noise)
        self.prior_prec = 0.1
        self.laplace = Laplace(self.model, self.prior_prec, self.lh)

    def standardize_data(self, y):
        self.standard_mean = y.mean()
        self.standard_std = y.std()
        return (y-self.standard_mean)/self.standard_std

    def detstandardize_data(self, y):
        return y*self.standard_std + self.standard_mean

    def apply_normalization(self, x):
        self.norm_factors = x.max(axis=0)
        return x/self.norm_factors

    def to_tensor(self, data):
        return torch.from_numpy(data).float()

    def to_numpy(self, data):
        return data.detach().numpy()

    def fit(self, X, Y):
        X = self.to_tensor(self.normalize_data(X))
        Y = self.to_tensor(self.standardize_data(Y))
        lr = 5e-3
        optim = Adam(self.model.parameters(), lr=lr)
        losses = list()
        n_epochs = 1000
        for i in range(n_epochs):
            f = self.model(X)
            w = parameters_to_vector(self.model.parameters())
            reg = 0.5 * self.prior_prec * w @ w
            loss = - self.lh.log_likelihood(Y, f) + reg
            loss.backward()
            optim.step()
            losses.append(loss.item())
            self.model.zero_grad()

        self.data = [(X, Y)]

    def infer(self):
        self.laplace.infer(self.data, cov_type='kron', dampen_kron=False)

    def predict(self, X)