import torch
import torch.nn as nn
import torchvision.models as models

def model(device):
    dense = models.densenet121(weights = models.DenseNet121_Weights.DEFAULT)
    dense.classifier = nn.Linear(dense.classifier.in_features, 2).to(device)
    for name, param in dense.features.named_parameters():
        if "denseblock4" not in name:
            param.requires_grad = False         #requires_grad=False -> freeze the parameters
    
    for module in dense.features.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    dense.to(device)
    optimizer = torch.optim.Adam([
        {'params': dense.features.denseblock4.parameters(), 'lr': 1e-4},
        {'params': dense.classifier.parameters(), 'lr': 1e-3}
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    return dense, optimizer, criterion, scheduler