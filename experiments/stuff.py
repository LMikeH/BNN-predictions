import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import parameters_to_vector

from laplace import Laplace

n_epochs = 10000
  # full batch
# true_sigma_noise = 0.3
# torch.manual_seed(711)

noise = 0.01
ndata = 20
batch_size = ndata


def f(x):
    return np.cos(x/12)*np.sin(x/9) + 2*np.exp(-np.abs(x-20)/3) 


def data_gen(n):
    xdata = np.sort(100*np.random.rand(n))
    ydata = f(xdata) + np.random.normal(0, noise, size=xdata.shape)

    return xdata.reshape(-1, 1), ydata.reshape(-1, 1)

X_train, y_train = data_gen(ndata)
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

# create simple sinusoid data set
# X_train = (torch.rand(150) * 8).unsqueeze(-1)
# y_train = torch.sin(X_train) + torch.randn_like(X_train) * true_sigma_noise
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
# X_test = torch.linspace(-5, 13, 500).unsqueeze(-1)  # +-5 on top of the training X-range
X_test = np.linspace(0, 100, 1000)
X_test = torch.from_numpy(X_test.reshape(-1,1)).float()

# create and train MAP model
model = torch.nn.Sequential(torch.nn.Linear(1, 100),
                            torch.nn.Tanh(),
                            torch.nn.Linear(100, 50),
                            torch.nn.Tanh(),
                            torch.nn.Linear(50, 25),
                            torch.nn.Tanh(),
                            torch.nn.Linear(25, 1))

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
w = parameters_to_vector(model.parameters())
for i in range(n_epochs):
    for X, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X), y) #+ 1e-2 * w @ w
        loss.backward()
        optimizer.step()

la = Laplace(model, 'regression', subset_of_weights='all', hessian_structure='kron')
la.fit(train_loader)
log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
for i in range(n_epochs):
    hyper_optimizer.zero_grad()
    neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
    neg_marglik.backward()
    hyper_optimizer.step()
print('sigma:', log_sigma.exp().item(), '; prior precision:', log_prior.exp().item())

x = X_test.flatten().cpu().numpy()
f_mu, f_var = la(X_test)
f_mu = f_mu.squeeze().detach().cpu().numpy()
f_sigma = f_var.squeeze().sqrt().cpu().numpy()
pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item()**2)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True,
                               figsize=(4.5, 2.8))
ax1.set_title('MAP')
ax1.scatter(X_train.flatten(), y_train.flatten(), alpha=0.7, color='tab:orange')
ax1.plot(x, f_mu, color='black', label='$f_{MAP}$')
ax1.legend()

ax2.set_title('LA')
ax2.scatter(X_train.flatten(), y_train.flatten(), alpha=0.7, color='tab:orange')
ax2.plot(x, f_mu, label='$\mathbb{E}[f]$')
ax2.fill_between(x, f_mu-pred_std*2, f_mu+pred_std*2, 
                 alpha=0.3, color='tab:blue', label='$2\sqrt{\mathbb{V}\,[f]}$')
ax2.legend()
ax1.set_ylim([-4, 6])
ax1.set_xlim([x.min(), x.max()])
ax2.set_xlim([x.min(), x.max()])
ax1.set_ylabel('$y$')
ax1.set_xlabel('$x$')
ax2.set_xlabel('$x$')
plt.tight_layout()
plt.savefig('docs/regression_example.png', dpi=300)
