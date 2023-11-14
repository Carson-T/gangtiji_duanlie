import torch
# from torch.cuda.amp import autocast
import torch.nn.functional as F
from tqdm import *


# training
def train(train_loader, model, criterion, optimizer, args):
    model.train()
    training_loss = 0.0
    for i, (img_list, targets) in enumerate(tqdm(train_loader)):
        if args["is_concat"]:
            img_list = img_list.to(args["device"])
        else:
            img_list = [img.to(args["device"]) for img in img_list]
        targets = targets.to(args["device"])
        # with autocast():
        output = model(img_list)
        loss = criterion(output, targets)
        # _, preds = torch.max(output, dim=1)
        # preds = output[:, 0]
        training_loss += loss.item() * len(targets)
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
            if args["is_concat"]:
                img_list = img_list.to(args["device"])
            else:
                img_list = [img.to(args["device"]) for img in img_list]
            targets = targets.to(args["device"])
            # with autocast():
            output = model(img_list)
            loss = criterion(output, targets)
            # _, preds = torch.max(output, dim=1)
            # preds = output[:, 0]
            val_loss += loss.item() * len(targets)
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
            if args["is_concat"]:
                img_list = img_list.to(args["device"])
            else:
                img_list = [img.to(args["device"]) for img in img_list]
            targets = targets.to(args["device"])
            # with autocast():
            output = model(img_list)
            loss = criterion(output, targets)
            # _, preds = torch.max(output, dim=1)
            # preds = output[:, 0]
            test_loss += loss.item() * len(targets)
            if i == 0:
                all_outputs = output
                all_targets = targets
            else:
                all_outputs = torch.cat((all_outputs, output))
                all_targets = torch.cat((all_targets, targets))
    all_outputs = F.softmax(all_outputs, dim=1)

    return all_outputs.cpu().detach(), all_targets.cpu().detach(), test_loss


def multi_task_train(train_loader1, train_loader2, model, criterion1, criterion2, optimizer, args):
    model.train()
    training_loss1 = training_loss2 = 0.0
    for i, (img_list1, targets1) in enumerate(tqdm(train_loader1)):
        img_list1 = [img.to(args["device"]) for img in img_list1]
        targets1 = targets1.to(args["device"])
        # with autocast():
        output1 = model(img_list1, "duanlie")
        loss1 = criterion1(output1, targets1)
        # _, preds = torch.max(output, dim=1)
        # preds = output[:, 0]
        training_loss1 += loss1.item() * len(targets1)
        if i == 0:
            all_outputs1 = output1
            all_targets1 = targets1
        else:
            all_outputs1 = torch.cat((all_outputs1, output1))
            all_targets1 = torch.cat((all_targets1, targets1))
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

    for i, (img_list2, targets2) in enumerate(tqdm(train_loader2)):
        img_list2 = [img.to(args["device"]) for img in img_list2]
        targets2 = targets2.to(args["device"])
        # with autocast():
        output2 = model(img_list2, "side")
        loss2 = criterion2(output2, targets2)
        # _, preds = torch.max(output, dim=1)
        # preds = output[:, 0]
        training_loss2 += loss2.item() * len(targets2)
        if i == 0:
            all_outputs2 = output2
            all_targets2 = targets2
        else:
            all_outputs2 = torch.cat((all_outputs2, output2))
            all_targets2 = torch.cat((all_targets2, targets2))
        optimizer.zero_grad()
        loss2.backward()
        optimizer.step()

    all_outputs1 = F.softmax(all_outputs1, dim=1)
    all_outputs2 = F.softmax(all_outputs2, dim=1)

    return ([all_outputs1.cpu().detach(), all_outputs2.cpu().detach()],
            [all_targets1.cpu().detach(), all_targets2.cpu().detach()],
            [training_loss1, training_loss2])


def multi_task_val(val_loader1, val_loader2, model, criterion1, criterion2, args):
    model.train()
    val_loss1 = val_loss2 = 0.0
    with torch.no_grad():
        for i, (img_list1, targets1) in enumerate(tqdm(val_loader1)):
            img_list1 = [img.to(args["device"]) for img in img_list1]
            targets1 = targets1.to(args["device"])
            # with autocast():
            output1 = model(img_list1, "duanlie")
            loss1 = criterion1(output1, targets1)
            # _, preds = torch.max(output, dim=1)
            # preds = output[:, 0]
            val_loss1 += loss1.item() * len(targets1)
            if i == 0:
                all_outputs1 = output1
                all_targets1 = targets1
            else:
                all_outputs1 = torch.cat((all_outputs1, output1))
                all_targets1 = torch.cat((all_targets1, targets1))

        for i, (img_list2, targets2) in enumerate(tqdm(val_loader2)):
            img_list2 = [img.to(args["device"]) for img in img_list2]
            targets2 = targets2.to(args["device"])
            # with autocast():
            output2 = model(img_list2, "side")
            loss2 = criterion2(output2, targets2)
            # _, preds = torch.max(output, dim=1)
            # preds = output[:, 0]
            val_loss2 += loss2.item() * len(targets2)
            if i == 0:
                all_outputs2 = output2
                all_targets2 = targets2
            else:
                all_outputs2 = torch.cat((all_outputs2, output2))
                all_targets2 = torch.cat((all_targets2, targets2))

        all_outputs1 = F.softmax(all_outputs1, dim=1)
        all_outputs2 = F.softmax(all_outputs2, dim=1)

    return ([all_outputs1.cpu().detach(), all_outputs2.cpu().detach()],
            [all_targets1.cpu().detach(), all_targets2.cpu().detach()],
            [val_loss1, val_loss2])