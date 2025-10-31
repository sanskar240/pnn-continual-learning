# src/pnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PNNColumn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.net(x)

class PNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=2, num_tasks=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.columns = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self.num_tasks = 0
        for _ in range(num_tasks):
            self.add_column(input_size, hidden_size, output_size)

    def add_column(self, input_size, hidden_size, output_size):
        in_dim = input_size if self.num_tasks == 0 else hidden_size * 2
        col = PNNColumn(in_dim, hidden_size, output_size)
        self.columns.append(col)
        for _ in range(self.num_tasks):
            self.adapters.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU()
            ))
        self.num_tasks += 1

    def forward(self, x, task_id):
        task_id = int(task_id.item()) if torch.is_tensor(task_id) else task_id
        col_input = x
        adapter_idx = 0
        for col_id in range(task_id + 1):
            col_out = self.columns[col_id](col_input)
            if col_id > 0:
                prev_out = self.columns[col_id-1](x).detach()
                for _ in range(col_id):
                    adapted = self.adapters[adapter_idx](prev_out)
                    col_input = torch.cat([col_input, adapted], dim=1)
                    adapter_idx += 1
            else:
                col_input = col_out
        return F.log_softmax(col_out, dim=1)