import torch
import torch.nn as nn
import torchvision.models as models

#def model(device):
#    dense = models.densenet121(weights = models.DenseNet121_Weights.DEFAULT)
#    dense.classifier = nn.Linear(dense.classifier.in_features, 2).to(device)
#    for name, param in dense.features.named_parameters():
#        if "denseblock4" not in name:
#            param.requires_grad = False         #requires_grad=False -> freeze the parameters
#    
#    for module in dense.features.modules():
#        if isinstance(module, nn.BatchNorm2d):
#            module.eval()
#    dense.to(device)
#    optimizer = torch.optim.Adam([
#        {'params': dense.features.denseblock4.parameters(), 'lr': 1e-4},
#        {'params': dense.classifier.parameters(), 'lr': 1e-3}
#    ])
#
#    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
#    criterion = nn.CrossEntropyLoss()
#    
#    return dense, optimizer, criterion, scheduler



def build_model(device, 
                lr_blocks, 
                lr_classifier, 
                freeze_strategy="classifier_only"):

    dense = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    dense.classifier = nn.Linear(dense.classifier.in_features, 2)

    # --- 1. FREEZE EVERYTHING ---
    for param in dense.features.parameters():
        param.requires_grad = False

    # --- 2. APPLY STRATEGY ---
    unfreeze_block3 = False
    unfreeze_block4 = False

    if freeze_strategy == "classifier_only":
        pass  # nothing else to unfreeze

    elif freeze_strategy == "last_block":
        unfreeze_block4 = True

    elif freeze_strategy == "two_last_blocks":
        unfreeze_block3 = True
        unfreeze_block4 = True

    else:
        raise ValueError(f"Unknown freeze strategy: {freeze_strategy}")

    # Actually unfreeze blocks
    if unfreeze_block3:
        for p in dense.features.denseblock3.parameters():
            p.requires_grad = True

    if unfreeze_block4:
        for p in dense.features.denseblock4.parameters():
            p.requires_grad = True

    # BatchNorm always eval
    for module in dense.features.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    dense.to(device)

    # --- 3. OPTIMIZER PARAM GROUPS ---
    params_to_optimize = []

    # Always optimize classifier
    params_to_optimize.append({
        'params': dense.classifier.parameters(),
        'lr': lr_classifier
    })

    # Add block3 + block4 under SAME lr if they are unfrozen
    block_params = []

    if unfreeze_block3:
        block_params.extend(list(dense.features.denseblock3.parameters()))

    if unfreeze_block4:
        block_params.extend(list(dense.features.denseblock4.parameters()))

    if len(block_params) > 0:
        params_to_optimize.append({
            'params': block_params,
            'lr': lr_blocks
        })

    optimizer = torch.optim.Adam(params_to_optimize)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    return dense, optimizer, criterion, scheduler