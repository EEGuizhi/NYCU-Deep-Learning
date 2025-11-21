import torch
import torch.nn as nn
import copy
from resnet20_int8 import QuantizedCifarResNet 
import torchvision
from torchvision import datasets, transforms, models
import argparse

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010))
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_acc(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    return acc

def test(path):
    try:
        state_dict = torch.load(path, weights_only=True)
    except:
        return False, "Failed to load your model! This may due to your file doesn't exist or your file isn't a state_dict."
    model = QuantizedCifarResNet()
    try:
        model.load_state_dict(state_dict)
    except:
        return False, "Your model's architecture seems to be wrong!"
    model.to(device)
    acc = test_acc(model)
    if acc < 90.0:
        return False, f"Your test accuracy is {acc:.2f}%, which is too low!"
    return True, f"Your model's test accuracy is {acc:.2f}%."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser")
    parser.add_argument('--path', type=str, required=True, help='Path to your pruned model.')
    args = parser.parse_args()

    result, msg = test(f"{args.path}")
    if result:
        print(f"Congratulations! You've achieved the goals of this task.")
    else:
        print(f"Oops! Something is wrong!")
    print(msg)