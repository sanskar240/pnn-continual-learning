# Progressive Neural Networks (PNN) – Continual Learning Demo

**Zero Forgetting. 100% on Moons after learning Circles.**

<img width="1000" height="400" alt="decision_boundary_plots" src="https://github.com/user-attachments/assets/ab7df5e9-befe-4f55-8eee-868da9ad00bb" />



---

## Tasks

| Task | Dataset | Description |
|------|--------|-------------|
| **0** | `make_moons(noise=0.1)` | Two interleaving half-circles |
| **1** | `make_circles(noise=0.1, factor=0.5)` | Two concentric circles |

---

## Training Log 

```text
Training Task 0 (Moons)...
Epoch 20 - Loss: 0.0538
Epoch 40 - Loss: 0.0000

Training Task 1 (Circles)...
Epoch 20 - Loss: 0.0009
Epoch 40 - Loss: 0.0001
