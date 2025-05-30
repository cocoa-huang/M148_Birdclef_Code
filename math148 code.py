import numpy as np
import pandas as pd
import torch
import ast
import torch.nn as nn
import torchvision.models as models
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def load_mel_specs_data(file_name):  
    data = np.load(file_name, allow_pickle=True)
    
    primary_labels = [entry[0] for entry in data[:100]] ## ONLY USING THE LAST 100 SAMPLES, change this number here and in the other two places if you want more
    secondary_labels = [entry[1] for entry in data[:100]] ## ONLY USING THE LAST 100 SAMPLES
    stripped_secondary_labels = [list(filter(None, ast.literal_eval(item))) for item in secondary_labels]
    
    combined_labels = [[] for _ in range(len(primary_labels))]
    for i in range(len(primary_labels)):
        combined_labels[i].append(primary_labels[i])
        for other_label in stripped_secondary_labels[i]:
            combined_labels[i].append(other_label)
    
    spectrograms = [entry[-1] for entry in data[:100]] ## ONLY USING THE LAST 100 SAMPLES
    spectrograms_np = np.stack(spectrograms)  # (N, 256, 256)
    spectrograms_np = spectrograms_np[:, np.newaxis, :, :]  # (N, 1, 256, 256)

    input_tensor = torch.tensor(spectrograms_np, dtype=torch.float32)
    
    return input_tensor, combined_labels

class BirdResNet(nn.Module):
    def __init__(self, num_classes=206):
        super(BirdResNet, self).__init__()
        self.base_model = models.resnet18(weights=True)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)

if __name__ == '__main__':
    # Load data
    input_tensor, combined_labels = load_mel_specs_data('mel_specs.npy')

    # Create label index mapping
    all_unique_labels = sorted(set(label for sublist in combined_labels for label in sublist))
    label_to_index = {label: idx for idx, label in enumerate(all_unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    num_classes = len(label_to_index)

    # Convert labels to multi-hot format
    label_indices = [[label_to_index[label] for label in sample] for sample in combined_labels]
    mlb = MultiLabelBinarizer(classes=list(range(num_classes)))
    multi_hot_labels = mlb.fit_transform(label_indices)
    label_tensor = torch.tensor(multi_hot_labels, dtype=torch.float32)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        input_tensor, label_tensor, test_size=0.2, random_state=42
    )

    # Initialize model, loss, optimizer
    model = BirdResNet(num_classes=num_classes)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    model.train()
    epochs = 10
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        binary_preds = (preds > 0.80)

        # Convert predictions to label indices
        pred_indices = [torch.nonzero(row, as_tuple=True)[0] for row in binary_preds]
        true_indices = [torch.nonzero(row, as_tuple=True)[0] for row in y_test]

        # Exact match
        matches = [
            set(pred.tolist()) == set(true.tolist())
            for pred, true in zip(pred_indices, true_indices)
        ]
        exact_match_accuracy = sum(matches) / len(matches)
        print(f"Exact Match Accuracy: {exact_match_accuracy:.4f}")
        
        # At least one correct label
        any_match = [
            len(set(pred.tolist()) & set(true.tolist())) > 0
            for pred, true in zip(pred_indices, true_indices)
        ]
        any_match_accuracy = sum(any_match) / len(any_match)
        print(f"At Least One Match Accuracy: {any_match_accuracy:.4f}")

        # Multi-label metrics
        y_true = y_test.numpy()
        y_pred = binary_preds.numpy()
        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        f1 = f1_score(y_true, y_pred, average='micro')

        print(f"Micro Precision: {precision:.4f}")
        print(f"Micro Recall:    {recall:.4f}")
        print(f"Micro F1 Score:  {f1:.4f}")