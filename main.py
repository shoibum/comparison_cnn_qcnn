import numpy as np
import time
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

import medmnist
from medmnist import INFO

import pennylane as qml
from pennylane import numpy as pnp

DATASET_NAME = 'pathmnist'
info = INFO[DATASET_NAME]
N_CHANNELS = info['n_channels']
N_CLASSES = len(info['label'])
TASK = info['task']

print(f"Dataset: {DATASET_NAME}, Task: {TASK}, Channels: {N_CHANNELS}, Classes: {N_CLASSES}")

# Data Transformations
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Load Data 
train_dataset = getattr(medmnist, info['python_class'])(
    split='train', transform=data_transform, download=True
)
val_dataset = getattr(medmnist, info['python_class'])(
    split='val', transform=data_transform, download=True
)
test_dataset = getattr(medmnist, info['python_class'])(
    split='test', transform=data_transform, download=True
)

# Create DataLoaders
BATCH_SIZE = 128
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

# Classical CNN
class ClassicalCNN(nn.Module):
    def __init__(self, n_channels=3, n_classes=9):
        super(ClassicalCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14x14
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=20 * 7 * 7, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=n_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output

# Quantum Layer Setup
N_QUBITS = 4 # Number of qubits
Q_LAYERS = 2 # Number of layers in the quantum circuit's PQC

try:
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    print(f"\nUsing PennyLane 'lightning.qubit' device with {N_QUBITS} qubits.")
except ImportError:
    dev = qml.device("default.qubit", wires=N_QUBITS)
    print(f"\nUsing PennyLane 'default.qubit' device with {N_QUBITS} qubits.")

# Define the Quantum Circuit (Quantum Node)
@qml.qnode(dev, interface='torch', diff_method='parameter-shift')
def quantum_circuit(inputs, weights):
    """Quantum circuit for feature processing."""
    # Angle encoding for input features
    angles = torch.atan(inputs) * 2
    for i in range(N_QUBITS):
        qml.RX(angles[i], wires=i)

    # Parameterized Quantum Circuit layers
    n_layers = weights.shape[0]
    for layer in range(n_layers):
        for i in range(N_QUBITS):
            qml.RZ(weights[layer, i], wires=i)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

    # Measurement: Expectation value of Pauli Z on each qubit
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(N_QUBITS)]

# Hybrid QCNN
class HybridQCNN(nn.Module):
    def __init__(self, n_channels=3, n_classes=9, n_qubits=N_QUBITS, q_layers=Q_LAYERS):
        super(HybridQCNN, self).__init__()
        self.n_qubits = n_qubits
        self.q_layers = q_layers

        self.classical_layer = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=self.n_qubits, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14x14
            nn.MaxPool2d(kernel_size=7, stride=7)  # 2x2
        )
        classical_output_size = self.n_qubits * 2 * 2

        self.downscale_fc = nn.Linear(classical_output_size, self.n_qubits)

        # Quantum layer
        weight_shapes = {"weights": (self.q_layers, self.n_qubits)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        self.classifier = nn.Linear(in_features=self.n_qubits, out_features=n_classes)

    def forward(self, x):
        x = self.classical_layer(x)
        x = x.view(x.shape[0], -1) 
        x = self.downscale_fc(x)   
        x = self.quantum_layer(x)  
        output = self.classifier(x)
        return output

# Utility Functions
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5):
    model.to(device)
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    model_name = model.__class__.__name__
    print(f"\n--- Training {model_name} ---")
    print(f"Parameters: {count_parameters(model)}")
    print(f"Epochs: {num_epochs}, Batch Size: {BATCH_SIZE}, Device: {device}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        start_time = time.time()

        for i, (inputs, labels) in enumerate(train_loader):
            labels = labels.squeeze().long()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                labels = labels.squeeze().long()
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s | "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc

    print(f"Finished Training {model_name}. Best Validation Accuracy: {best_val_acc:.4f}")
    return history

# Evaluation Function
def evaluate_model(model, test_loader, criterion, device):
    model.to(device)
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    start_time = time.time()
    model_name = model.__class__.__name__

    print(f"\n--- Evaluating {model_name} on Test Set ---")

    with torch.no_grad():
        for inputs, labels in test_loader:
            labels = labels.squeeze().long()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    final_loss = test_loss / len(test_loader.dataset)
    final_acc = correct_test / total_test
    eval_time = time.time() - start_time

    print(f"Evaluation Time: {eval_time:.2f}s")
    print(f"Test Loss: {final_loss:.4f}")
    print(f"Test Accuracy: {final_acc:.4f}")
    return final_loss, final_acc

if __name__ == "__main__":
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 30 # Increase this for better results
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Common Components
    criterion = nn.CrossEntropyLoss()

    # Train and Evaluate Classical CNN
    cnn_model = ClassicalCNN(n_channels=N_CHANNELS, n_classes=N_CLASSES)
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
    cnn_history = train_model(cnn_model, train_loader, val_loader, criterion, cnn_optimizer, DEVICE, num_epochs=NUM_EPOCHS)
    cnn_test_loss, cnn_test_acc = evaluate_model(cnn_model, test_loader, criterion, DEVICE)

    # Train and Evaluate Hybrid QCNN
    qcnn_model = HybridQCNN(n_channels=N_CHANNELS, n_classes=N_CLASSES, n_qubits=N_QUBITS, q_layers=Q_LAYERS)
    qcnn_optimizer = optim.Adam(qcnn_model.parameters(), lr=LEARNING_RATE)
    qcnn_history = train_model(qcnn_model, train_loader, val_loader, criterion, qcnn_optimizer, DEVICE, num_epochs=NUM_EPOCHS)
    qcnn_test_loss, qcnn_test_acc = evaluate_model(qcnn_model, test_loader, criterion, DEVICE)

    # Final Comparison Summary
    print("\n--- Comparison Summary ---")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}")
    print("-" * 30)
    print(f"Classical CNN:")
    print(f"  Parameters: {count_parameters(cnn_model)}")
    print(f"  Test Accuracy: {cnn_test_acc:.4f}")
    print(f"  Test Loss: {cnn_test_loss:.4f}")
    print("-" * 30)
    print(f"Hybrid QCNN:")
    print(f"  Parameters: {count_parameters(qcnn_model)}")
    print(f"  Test Accuracy: {qcnn_test_acc:.4f}")
    print(f"  Test Loss: {qcnn_test_loss:.4f}")
    print("-" * 30)

    # Plotting Training History
    epochs_range = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, cnn_history['val_acc'], 'bo-', label='CNN Validation Accuracy')
    plt.plot(epochs_range, qcnn_history['val_acc'], 'ro-', label='QCNN Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(epochs_range)
    plt.legend()
    plt.grid(True)
    plt.savefig('cnn_vs_qcnn_val_accuracy.png')
    print("\nValidation accuracy plot saved as 'cnn_vs_qcnn_val_accuracy.png'")
    plt.show()