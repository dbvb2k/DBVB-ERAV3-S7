from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Train Phase transformations
train_transforms = transforms.Compose([
                   # transforms.Resize((28, 28)),
                   # transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                   transforms.RandomRotation((-7.0, 7.0), fill=(1,)),    
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
                   # Note the difference between (0.1307) and (0.1307,)
                   ])

# Test Phase transformations
test_transforms = transforms.Compose([
                   # transforms.Resize((28, 28)),
                   # transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
                   ])

train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

SEED = 5
BATCH_SIZE = 128
CUDA_BATCH_SIZE = 64

# Use CUDA if available
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

# Dataloader arguments - something you'll fetch these from cmdprmt
dataloader_args = dict(shuffle=True, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=CUDA_BATCH_SIZE)

# Train dataloader
train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

# Test dataloader
test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

# This notebook experiment uses Net2() model in Models.py
from Models import get_model2, save_model2

from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
model = get_model2().to(device)
summary(model, input_size=(1, 28, 28))

from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # Get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Training: Batch_id={batch_idx} Loss={loss.item()} Accuracy={100 * correct / processed:0.2f}')
    train_acc.append(100 * correct / processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    accuracy = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    test_acc.append(accuracy)
    return accuracy  # Return the accuracy for the scheduler

# Run the model 2
model = get_model2().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Initialize ReduceLROnPlateau scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',           # Since we're tracking validation accuracy
    factor=0.1,          # Reduce LR by factor of 10
    patience=3,          # Number of epochs with no improvement after which LR will be reduced
    verbose=True,        # Print message when LR is reduced
    min_lr=1e-6,        # Minimum LR
    threshold=0.001,     # Minimum change to qualify as an improvement
    threshold_mode='rel' # Relative change
)

EPOCHS = 15
for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train(model, device, train_loader, optimizer, epoch)
    accuracy = test(model, device, test_loader)  # Get the accuracy
    scheduler.step(accuracy)  # Pass the accuracy to scheduler

# Plotting the results

t = [t_items.item() for t_items in train_losses]

fig, axs = plt.subplots(2,2,figsize=(15, 10))
axs[0, 0].plot(t)
axs[0, 0].set_title("Training Loss")
axs[1, 0].plot(train_acc)
axs[1, 0].set_title("Training Accuracy")
axs[0, 1].plot(test_losses)
axs[0, 1].set_title("Test Loss")
axs[1, 1].plot(test_acc)
axs[1, 1].set_title("Test Accuracy")