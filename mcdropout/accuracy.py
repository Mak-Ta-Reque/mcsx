
from xml.parsers.expat import model
import torch
from torch.utils.data import DataLoader, TensorDataset


def _get_model_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device('cpu')


def _mc_acc(model, x_test, y_test, num_samples):
    model.train()  # Set the model to training mode to enable dropout
    device = _get_model_device(model)
    correct = 0
    total = 0

    outputs_list = []
     # Perform forward pass num_samples times to simulate dropout
    for i in range(num_samples):
        outputs = model(x_test.to(device))
        outputs_list.append(outputs.unsqueeze(0))

    # Stack outputs, take mean over num_samples dimension
    outputs_mean = torch.cat(outputs_list).mean(dim=0)

    # Calculate predictions
    _, predicted = torch.max(outputs_mean.data, 1)

    total += y_test.size(0)
    correct += (predicted == y_test.to(device)).sum().item()

    accuracy = 100 * correct / total
    model.eval()
    return accuracy

def mc_acc(model, x_test, y_test, num_samples, batch_size = 5):
    model.train()
    device = _get_model_device(model)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
        
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs_list = []

            for i in range(num_samples):
                outputs = model(inputs)
                outputs_list.append(outputs.unsqueeze(0))

            outputs_mean = torch.cat(outputs_list).mean(dim=0)
            _, predicted = torch.max(outputs_mean.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total 
    model.eval()
    return accuracy

def acc(model, x_test, y_test, batch_size = 20):
    model.eval()  # Set the model to evaluation mode
    device = _get_model_device(model)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    model.eval()
    return accuracy

def cfn_acc(model, x_test, y_test, batch_size = 20):
    model.train()  # Ensure dropout is active
    device = _get_model_device(model)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    model.eval()
    return accuracy

