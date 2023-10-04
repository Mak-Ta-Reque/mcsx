
import torch


def mc_acc(model, x_test, y_test, num_samples):
    model.train()  # Set the model to training mode to enable dropout
    correct = 0
    total = 0

    outputs_list = []
     # Perform forward pass num_samples times to simulate dropout
    for i in range(num_samples):
        outputs = model(x_test)
        outputs_list.append(outputs.unsqueeze(0))

    # Stack outputs, take mean over num_samples dimension
    outputs_mean = torch.cat(outputs_list).mean(dim=0)

    # Calculate predictions
    _, predicted = torch.max(outputs_mean.data, 1)

    total += y_test.size(0)
    correct += (predicted == y_test).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def acc(model, x_test, y_test):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        #x_test = x_test.to(device)
        #y_test = y_test.to(device)

        outputs = model(x_test)

        # Calculate predictions
        _, predicted = torch.max(outputs.data, 1)

        total += y_test.size(0)
        correct += (predicted == y_test).sum().item()

    accuracy = 100 * correct / total
    return accuracy
    return accuracy
