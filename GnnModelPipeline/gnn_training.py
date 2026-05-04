import torch.nn as nn
from torch_geometric.nn import SAGEConv
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


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



def evaluate_model(model, test_data):
    model.eval()

    device = next(model.parameters()).device
    test_data = test_data.to(device)

    with torch.no_grad():
        out = model(test_data)
        probs = torch.softmax(out, dim=1)
        preds = out.argmax(dim=1)

        y_true = test_data.y.cpu().numpy()
        y_pred = preds.cpu().numpy()
        y_prob = probs.cpu().numpy()

    # -----------------------------
    # Metrics
    # -----------------------------
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # ROC-AUC (only if binary or one-vs-rest works cleanly)
    try:
        if y_prob.shape[1] == 2:
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except Exception:
        roc_auc = None

    # -----------------------------
    # Print results
    # -----------------------------
    print("\n📊 Evaluation Results")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    if roc_auc is not None:
        print(f"ROC-AUC   : {roc_auc:.4f}")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }