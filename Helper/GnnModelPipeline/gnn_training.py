import torch.nn as nn
from torch_geometric.nn import SAGEConv
import torch

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.dropout(x)

        return self.classifier(x)

def train_model(data):
    model = GNNModel(
        input_dim=data.x.shape[1],
        hidden_dim=32,
        num_classes=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(50):
        optimizer.zero_grad()

        out = model(data)
        loss = loss_fn(out, data.y)

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            pred = out.argmax(dim=1)
            acc = (pred == data.y).float().mean()

            print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    return model