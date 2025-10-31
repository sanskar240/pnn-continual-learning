# src/train.py
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from pnn import PNN
from utils import get_task_data, plot_boundary
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    model = PNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    os.makedirs("results", exist_ok=True)

    print("Training Task 0: Moons")
    train_loader, _ = get_task_data(0)
    for epoch in range(50):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, 0)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1} - Loss: {loss.item():.4f}")

    print("Training Task 1: Circles")
    train_loader, _ = get_task_data(1)
    for epoch in range(50):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, 1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1} - Loss: {loss.item():.4f}")

    # Final Eval
    def eval(task_id):
        _, test_loader = get_task_data(task_id)
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x, task_id)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total

    acc0 = eval(0)
    acc1 = eval(1)
    print(f"\nFinal Accuracies:")
    print(f"Task 0 (Moons): {acc0:.4f}")
    print(f"Task 1 (Circles): {acc1:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    plot_boundary(model, 0, f"Task 0: Moons ({acc0:.1%})", ax1, device)
    plot_boundary(model, 1, f"Task 1: Circles ({acc1:.1%})", ax2, device)
    plt.tight_layout()
    plt.savefig("results/demo.png")
    plt.show()

    # Save model
    torch.save(model.state_dict(), "results/pnn_model.pth")
    print("Model saved to results/pnn_model.pth")

if __name__ == "__main__":
    main()