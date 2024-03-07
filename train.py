import os, json, time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from model import build_model
from datasets import train_loader, valid_loader, full_dataset
import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=20,
    help='number of epochs to train our network for')
parser.add_argument('-f', '--folder', type=str, default='default',
    help='folder name of the current training')
args = vars(parser.parse_args())

# learning_parameters 
lr = 1e-4
print(f"Learning rate: {lr}")
# model type
model_type = 1

epochs = args['epochs']

output_path = os.path.join('outputs', args['folder'])
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

# build the model, fine_tune=True
model = build_model(
    pretrained=True, fine_tune=True, num_classes=len(full_dataset.classes), model_type=model_type
).to(device)
# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.\n")
# optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)
# loss function
criterion = nn.CrossEntropyLoss()

# -------------------
'''
# use different weighted cross entropy loss while not upsampling the data
import sklearn.utils.class_weight as class_weight
import numpy as np
imbalanced_labels =len([i[1] for i in full_dataset.imgs])
class_weights=class_weight.compute_class_weight('balanced', classes=np.unique(imbalanced_labels), y=np.array(imbalanced_labels))
class_weights=torch.tensor(class_weights,dtype=torch.float)
criterion = nn.CrossEntropyLoss(class_weights)
'''
# -------------------

# some utils
matplotlib.style.use('ggplot')
def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                # }, os.path.join(output_path, 'best_model.pth'))
                }, os.path.join(output_path, f'model_epoch{epochs}.pth'))

def save_log(train_acc, valid_acc, train_loss, valid_loss, class_acc):
    """
    Function to save the loss and accuracy logs
    """
    training_logs = {
        'lr': lr,
        'epochs': epochs,
        'model_type': model_type,
        'train_acc': train_acc,
        'valid_acc': valid_acc,
        'train_loss': train_loss, 
        'valid_loss': valid_loss,
        'class_acc': class_acc,
    }
    
    with open(os.path.join(output_path, 'train.log'), mode='w') as f:
        f.write(json.dumps(training_logs, indent=2))
    
def save_plots(train_acc, valid_acc, train_loss, valid_loss, class_acc):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'accuracy.png'))
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'loss.png'))

    # class-wise accuracy plots
    plt.figure(figsize=(10, 7))
    gd_acc = [i[0] for i in class_acc]
    ngd_acc = [i[1] for i in class_acc]
    class_acc = [gd_acc, ngd_acc]
    for i in range(len(full_dataset.classes)):
        plt.plot(
            class_acc[i], linestyle='-', 
            label=f'{full_dataset.classes[i]}'
        )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'class_accuracy.png'))


# training
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # calculate the accuracy
        preds = torch.argmax(outputs, dim=1)
        train_running_correct += (preds == labels).sum().item()
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

# validation
def validate(model, testloader, criterion, class_names):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    
    # we need two lists to keep track of class-wise accuracy
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
            preds = torch.argmax(outputs, dim=1)
            valid_running_correct += (preds == labels).sum().item()
            
            # calculate the accuracy for each class
            correct  = (preds == labels).squeeze()
            for i in range(len(preds)):
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    
    class_acc = []
    for i in range(len(class_names)):
        class_acc.append(100 * class_correct[i] / class_total[i])
        
    return epoch_loss, epoch_acc, class_acc

# lists to keep track of losses and accuracies
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
class_acc = []
aver_acc_max = 0

# check output folder exists or not
if not os.path.exists('outputs'):
    os.mkdir('outputs')
if not os.path.exists(output_path):
    os.mkdir(output_path)
# start the training
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                              optimizer, criterion)
    valid_epoch_loss, valid_epoch_acc, class_epoch_acc = validate(model, valid_loader,  
                                                 criterion, full_dataset.classes)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    class_acc.append(class_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    # print the accuracy for each class after every epoch
    for i in range(len(full_dataset.classes)):
        print(f"Validation Accuracy of class {full_dataset.classes[i]}: {class_epoch_acc[i]}")
    print('-'*50)
    w = 5
    if aver_acc_max < (5 * valid_epoch_acc + train_epoch_acc) / 6 and epoch > 50:
        aver_acc_max = (5 * valid_epoch_acc + train_epoch_acc) / 6
        # save the trained model weights
        save_model(epoch, model, optimizer, criterion)
        print(f"Best model updated in EPOCH:{epoch}.")
    elif epoch > 50 and valid_epoch_acc > 0.9 and train_epoch_acc > 0.9:
        # save the trained model weights
        save_model(epoch, model, optimizer, criterion)
        # print(f"Best model updated in EPOCH:{epoch}.")

# save the loss and accuracy logs
save_log(train_acc, valid_acc, train_loss, valid_loss, class_acc)

# save the loss and accuracy plots
save_plots(train_acc, valid_acc, train_loss, valid_loss, class_acc)
print('TRAINING COMPLETE')

