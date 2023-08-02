import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
from tqdm import *



# training
def train(train_loader, model, criterion, optimizer, scaler, args):
    model.train()
    training_loss = 0.0
    for i, (images, targets) in enumerate(tqdm(train_loader)):
        images = images.to(args["device"])
        targets = targets.to(args["device"])
        # with autocast():
        output = model(images)
        loss = criterion(output, targets)
        # _, preds = torch.max(output, dim=1)
        # preds = output[:, 0]
        training_loss += loss.item()
        if i == 0:
            all_outputs = output
            all_targets = targets
        else:
            all_outputs = torch.cat((all_outputs, output))
            all_targets = torch.cat((all_targets, targets))
        
        all_outputs = F.softmax(all_outputs, dim=1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # optimizer.zero_grad()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()


    return all_outputs.cpu().detach(), all_targets.cpu().detach(), training_loss


def val(val_loader, model, criterion, args):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(val_loader)):
            images = images.to(args["device"])
            targets = targets.to(args["device"])
            # with autocast():
            output = model(images)
            loss = criterion(output, targets)
            # _, preds = torch.max(output, dim=1)
            # preds = output[:, 0]
            val_loss += loss.item()
            if i == 0:
                all_outputs = output
                all_targets = targets
            else:
                all_outputs = torch.cat((all_outputs, output))
                all_targets = torch.cat((all_targets, targets))
            all_outputs = F.softmax(all_outputs, dim=1)
            
    return all_outputs.cpu().detach(), all_targets.cpu().detach(), val_loss

def test(test_loader, model, criterion, args):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(test_loader)):
            images = images.to(args["device"])
            targets = targets.to(args["device"])
            # with autocast():
            output = model(images)
            loss = criterion(output, targets)
            # _, preds = torch.max(output, dim=1)
            # preds = output[:, 0]
            test_loss += loss.item()
            if i == 0:
                all_outputs = output
                all_targets = targets
            else:
                all_outputs = torch.cat((all_outputs, output))
                all_targets = torch.cat((all_targets, targets))
            all_outputs = F.softmax(all_outputs, dim=1)
            
    return all_outputs.cpu().detach(), all_targets.cpu().detach(), test_loss

def external_train(train_loader, model, criterion, optimizer, scaler, args):
    model.train()
    training_loss = 0.0
    for i, (images, targets, groups) in enumerate(tqdm(train_loader)):
        images = images.to(args["device"])
        targets = targets.to(args["device"])
        with autocast():
            output = model(images)
            loss = criterion(output, targets)
            # _, preds = torch.max(output, dim=1)
            preds = output[:, 0]
        training_loss += loss.item()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
