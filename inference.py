import torch, json, os
from datasets import ImbalancedScrewDataset, full_dataset, val_dataset
from sklearn.metrics import confusion_matrix, auc, precision_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import model
from tqdm.auto import tqdm
import numpy as np
import seaborn as sns

# ***********************
# remember to modify model path and model type every time you want to run inference
output_path = 'outputs/model1/'
model_name = "model_epoch56.pth"
model_path = os.path.join(output_path, model_name)
fig_path = 'analysis_figures'
model_type = 1
# ***********************

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filepath', type=str, default='ckpt/model_epoch56.pth',
    help='model path name of the ckpt file')
args = vars(parser.parse_args())

# make sure fig_path exists
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

# check whether trained model exists
if not os.path.exists(output_path):
    model_path = args['filepath']
else:
    if args['filepath'] != 'ckpt/model_epoch56.pth':
        model_path = args['filepath']

assert os.path.exists('ckpt/model_epoch56.pth'), 'default ckpt model does not exist'

# model_path = os.path.join(output_path, model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Current device: {}'.format(device))

# load checkpoint parameters
model = model.build_model(
    pretrained=False, fine_tune=False, num_classes=2
).to(device)
ckpt = torch.load(model_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])

with open('test_data.json', mode='r') as f:
    test_data = json.load(f)

# construct test dataset
test_dataset = test_data['good']
for k in test_data['not-good']:
    test_dataset.append(k)
    
classes = ['good', 'not-good']
test_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,), (0.5,)
    )
])

# use customized dataset
test_dataset = ImbalancedScrewDataset(test_dataset, classes=classes, transform=test_transform)
test_loader = DataLoader(
    test_dataset, batch_size = 1, shuffle = False, num_workers = 4
)
print('Test dataset size: {}'.format(len(test_dataset)))
print('Current model path: {}'.format(model_path))

valid_loader = DataLoader(
    val_dataset, batch_size = 1, shuffle = False, num_workers = 4
)

def validate(model, testloader, class_names, mode='test'):
    # model go to eval mode
    model.eval()
    print('Valid on the {} set'.format(mode))
    y_true = []
    y_pred = []

    total_correct = 0
    total_images = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            total_images += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(image)
            
            # calculate the accuracy
            y_true.append(labels.squeeze().item())
            y_pred.append(torch.argmax(outputs, dim=1).item())
            pred = torch.argmax(outputs, dim=1)
            correct  = (pred == labels).squeeze()
            total_correct += correct.item()

        print(f"Accuracy on the {mode} set: {100. * total_correct / total_images}, ({total_correct}/{total_images})")
    return y_true, y_pred

# inference on validation set
print('\n')
print('------- Inference START -------')
y_true, y_pred = validate(model, valid_loader, classes, mode='valid')

# confusion matrix
# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['good', 'not good'], yticklabels=['good', 'not good'])
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig(os.path.join(fig_path, 'confusion_matrix_valid.png'), dpi=300)
print('Confusion matrix saved to the model folder.')

# precision, recall, f1 score
print(f"Precision: {precision_score(y_true, y_pred)}")
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy_score(y_true, y_pred)}")

print('------- Inference END -------')
print('\n')

# inference on test set
print('\n')
print('------- Inference START -------')
y_true, y_pred = validate(model, test_loader, classes, mode='test')


# confusion matrix
# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['good', 'not good'], yticklabels=['good', 'not good'])
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig(os.path.join(fig_path, 'confusion_matrix_test.png'), dpi=300)
print('Confusion matrix saved to the model folder.')

# precision, recall, f1 score
print(f"Precision: {precision_score(y_true, y_pred)}")
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy_score(y_true, y_pred)}")

print('------- Inference END -------')
print('\n')