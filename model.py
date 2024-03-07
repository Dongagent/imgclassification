import torchvision.models as models
import torch.nn as nn
def build_model(pretrained=True, fine_tune=True, num_classes=1, model_type=1):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # model = models.resnet50(weights=models.ResNet50_Weights)
    elif not pretrained:
        print('[INFO]: Not loading pre-trained weights')
        model = models.resnet34()
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
            
    # change the final classification head, it is trainable
    
    # model 1
    if model_type == 1:
        model.fc = nn.Linear(512, num_classes)
    # model 2
    elif model_type == 2:
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    # model 3
    elif model_type == 3:
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    # model.fc = nn.Linear(2048, num_classes)
    
    return model