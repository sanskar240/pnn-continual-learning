# src/utils.py
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_task_data(task_id, n_samples=1000, batch_size=32):
    if task_id == 0:
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    elif task_id == 1:
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
    else:
        raise ValueError("Only tasks 0 and 1")
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(test_ds, batch_size=batch_size)

def plot_boundary(model, task_id, title, ax, device):
    model.eval()
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
    with torch.no_grad():
        Z = model(grid, task_id).argmax(1).cpu().numpy().reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.8, cmap='RdYlBu')
    _, test_loader = get_task_data(task_id)
    X_test, y_test = test_loader.dataset.tensors
    X_test, y_test = X_test.cpu().numpy(), y_test.cpu().numpy()
    ax.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='RdYlBu', edgecolor='k', s=20)
    ax.set_title(title)
    ax.set_xlim(-3,3); ax.set_ylim(-3,3)