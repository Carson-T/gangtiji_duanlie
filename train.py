import torch
# from torch.cuda.amp import autocast
import torch.nn.functional as F
from tqdm import *


# training
def train(train_loader, model, criterion, optimizer, args):
    model.train()
    training_loss = 0.0
    for i, (img_list, targets) in enumerate(tqdm(train_loader)):
        img_list = [img.to(args["device"]) for img in img_list]
        targets = targets.to(args["device"])
        # with autocast():
        output = model(img_list)
        loss = criterion(output, targets)
        # _, preds = torch.max(output, dim=1)
        # preds = output[:, 0]
        training_loss += loss.item()*len(targets)
        if i == 0:
            all_outputs = output
            all_targets = targets
        else:
            all_outputs = torch.cat((all_outputs, output))
            all_targets = torch.cat((all_targets, targets))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # optimizer.zero_grad()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
    all_outputs = F.softmax(all_outputs, dim=1)

    return all_outputs.cpu().detach(), all_targets.cpu().detach(), training_loss


def val(val_loader, model, criterion, args):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, (img_list, targets) in enumerate(tqdm(val_loader)):
            img_list = [img.to(args["device"]) for img in img_list]
            targets = targets.to(args["device"])
            # with autocast():
            output = model(img_list)
            loss = criterion(output, targets)
            # _, preds = torch.max(output, dim=1)
            # preds = output[:, 0]
            val_loss += loss.item()*len(targets)
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
        for i, (img_list, targets) in enumerate(tqdm(test_loader)):
            img_list = [img.to(args["device"]) for img in img_list]
            targets = targets.to(args["device"])
            # with autocast():
            output = model(img_list)
            loss = criterion(output, targets)
            # _, preds = torch.max(output, dim=1)
            # preds = output[:, 0]
            test_loss += loss.item()*len(targets)
            if i == 0:
                all_outputs = output
                all_targets = targets
            else:
                all_outputs = torch.cat((all_outputs, output))
                all_targets = torch.cat((all_targets, targets))
    all_outputs = F.softmax(all_outputs, dim=1)

    return all_outputs.cpu().detach(), all_targets.cpu().detach(), test_loss


