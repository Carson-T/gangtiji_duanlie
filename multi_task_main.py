import torch.optim.lr_scheduler
from torch.utils.data import DataLoader, sampler
from torch.utils.data.dataset import Subset
from torch.cuda.amp import GradScaler
import torch.backends.cudnn as cudnn
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
import json
import random
import collections
import timm
from train import *
from dataset import *
from argparser import args_parser
from models import *
from utils.loss import *
from utils.plot import plot_matrix
from utils.transform import *
from utils.function import *
import wandb


def set_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark = True
    # cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def kfold_split(train_val_dataset):
    train_loaders = []
    val_loaders = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args["random_seed"])
    for k, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset, train_val_dataset.labels)):
        train_dataset = Subset(train_val_dataset, train_idx)
        val_dataset = Subset(train_val_dataset, val_idx)

        train_loaders.append(DataLoader(train_dataset, batch_size=args["batch_size"],
                                        shuffle=False if args["sampler"] else True,
                                        num_workers=args["num_workers"],
                                        pin_memory=True, drop_last=False))
        val_loaders.append(DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False,
                                      num_workers=args["num_workers"], pin_memory=True, drop_last=False))

    return train_loaders, val_loaders


def main(args):
    # scaler = GradScaler()
    # train_transform, val_transform, test_transform = tv_transform(args)
    train_transform, val_transform, test_transform = at_transform(args)

    two_train_val_dataset = Two_Dataset(args["data_path"], train_transform, is_test=False)
    two_test_loader = DataLoader(Two_Dataset(args["data_path"], test_transform, is_test=True),
                                 batch_size=args["batch_size"], shuffle=False,
                                 num_workers=args["num_workers"],
                                 pin_memory=True, drop_last=False)
    counts = collections.Counter(two_train_val_dataset.labels)
    class_weight1 = (counts[0] + counts[1]) / torch.tensor([counts[0], counts[1]]).to(args["device"])

    three_train_val_dataset = Three_Dataset(args["data_path"], train_transform, is_test=False)
    three_test_loader = DataLoader(Three_Dataset(args["data_path"], test_transform, is_test=True),
                                   batch_size=args["batch_size"], shuffle=False,
                                   num_workers=args["num_workers"],
                                   pin_memory=True, drop_last=False)
    counts = collections.Counter(three_train_val_dataset.labels)
    class_weight2 = (counts[0] + counts[1] + counts[2]) / torch.tensor([counts[0], counts[1], counts[2]]).to(
        args["device"])

    print(counts)
    two_train_loaders, two_val_loaders = kfold_split(two_train_val_dataset)
    three_train_loaders, three_val_loaders = kfold_split(three_train_val_dataset)

    for fold in args["fold"]:
        # print("num of train_val_dataset:", len(train_val_dataset.labels))
        # print("num of train images:", len(train_loaders[fold - 1].dataset))
        # print("num of val images:", len(val_loaders[fold - 1].dataset))
        # print("num of test images:", len(test_loader.dataset))
        version_name = args["version_name"] + f"-fold{fold}"
        # writer = SummaryWriter(log_dir=args["log_dir"] + "/" + version_name)
        wandb.init(
            # set the wandb project where this run will be logged
            project="gangtiji-duanlie",
            name=version_name,
            # id=version_name,
            dir=args["log_dir"],
            resume=True if args["resume"] else False,
            # track hyperparameters and run metadata
            config=args
        )
        if args["model_source"] == "timm":
            if args["drop_path_rate"] > 0:
                pretrained_model = timm.create_model(args["backbone"], drop_rate=args["drop_rate"],
                                                     drop_path_rate=args["drop_path_rate"], pretrained=True)
            else:
                pretrained_model = timm.create_model(args["backbone"], drop_rate=args["drop_rate"], pretrained=True)
            if "resnet" in args["backbone"]:
                model = Resnet(pretrained_model, args["num_classes"])
            elif "efficientnet" in args["backbone"]:
                model = Efficientnet(pretrained_model, args["num_classes"])
            elif "convnext" in args["backbone"]:
                model = Convnext(pretrained_model, args["num_classes"])

        else:
            pretrained_model = models.get_model(name=args["backbone"], weights="DEFAULT", dropout=args["drop_rate"])
            if "efficientnet" in args["backbone"]:
                model = MultiTask_efficientnet_tv(pretrained_model, *args["num_classes"])

        head1, head2 = model.get_head()
        base_params = filter(
            lambda p: id(p) not in list(map(id, head1.parameters())) + list(map(id, head2.parameters())),
            model.parameters())
        groups_params = [{"params": base_params, "lr": args["lr"][0]},
                         {"params": head1.parameters(), "lr": args["lr"][1]},
                         {"params": head2.parameters(), "lr": args["lr"][2]}]
        print("models init finish")
        if args["is_parallel"] == 1:
            model = nn.DataParallel(model, device_ids=args["device_ids"])
        model.to(args["device"])
        if args["init"] == "xavier":
            model.apply(xavier)
        elif args["init"] == "kaiming":
            model.apply(kaiming)

        if args["pretrained_path"]:
            model.load_state_dict(torch.load(args["pretrained_path"]), strict=True)

        if args["optim"] == "AdamW":
            optimizer = torch.optim.AdamW(groups_params, weight_decay=args["weight_decay"])
        elif args["optim"] == "SGD":
            optimizer = torch.optim.SGD(groups_params, momentum=0.9, weight_decay=args["weight_decay"])

        if args["loss_func"] == "CEloss":
            loss_func1 = torch.nn.CrossEntropyLoss(weight=class_weight1 if args["use_weighted_loss"] == 1 else None).to(
                args["device"])
            loss_func2 = torch.nn.CrossEntropyLoss(weight=class_weight2 if args["use_weighted_loss"] == 1 else None).to(
                args["device"])
        # elif args["loss_func"] == "FocalLoss":
        #     loss_func = FocalLoss().to(args["device"])
        elif args["loss_func"] == "LabelSmoothLoss":
            loss_func1 = LabelSmoothLoss(weight=class_weight1).to(args["device"])
            loss_func2 = LabelSmoothLoss(weight=class_weight2).to(args["device"])

        if args["lr_scheduler"] == "Warm-up-Cosine-Annealing":
            init_ratio, warm_up_steps, min_lr_ratio, max_steps = args["init_ratio"], args["epochs"] / 10, args[
                "min_lr_ratio"], args["epochs"]
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step:
            (1 - init_ratio) / (warm_up_steps - 1) * step + init_ratio if step < warm_up_steps - 1
            else (1 - min_lr_ratio) * 0.5 * (np.cos(
                (step - (warm_up_steps - 1)) / (max_steps - (warm_up_steps - 1)) * np.pi) + 1) + min_lr_ratio)
        elif args["lr_scheduler"] == "StepLR":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args["step_size"], gamma=args["gamma"])

        best_test_auc = 0.0
        best_test_acc = 0.0
        best_epoch_metrics = []
        init_epoch = 1

        if args["resume"] != "":
            checkpoint = torch.load(args["resume"])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            init_epoch = checkpoint["epoch"] + 1
            best_test_auc = checkpoint["best_test_auc"]

        for iter in range(init_epoch, args["epochs"] + 1):
            train_outputs, train_targets, train_loss = multi_task_train(two_train_loaders[fold - 1], three_train_loaders[fold-1], model, loss_func1, loss_func2, optimizer, args)
            duanlie_train_acc, duanlie_train_auc, duanlie_train_auprc = calculate_metrics(train_outputs[0], train_targets[0], train_loss[0])
            side_train_acc, side_train_auc, side_train_auprc = calculate_metrics(train_outputs[1], train_targets[1], train_loss[1])
            if args["validation"] == 1:
                val_outputs, val_targets, val_loss = multi_task_val(two_val_loaders[fold - 1], three_val_loaders[fold-1], model, loss_func1, loss_func2, args)
                duanlie_val_acc, duanlie_val_auc, duanlie_val_auprc = calculate_metrics(val_outputs[0], val_targets[0], val_loss[0])
                side_val_acc, side_val_auc, side_val_auprc = calculate_metrics(val_outputs[1], val_targets[1], val_loss[1])
            # else:
            #     val_loss = 0.0
            #     val_targets = [0]
            #     val_acc, val_auc, val_auprc = 0.0, 0.0, 0.0
            test_outputs, test_targets, test_loss = multi_task_val(two_test_loader, three_test_loader, model, loss_func1, loss_func2, args)
            duanlie_test_acc, duanlie_test_auc, duanlie_test_auprc = calculate_metrics(test_outputs[0], test_targets[0], test_loss[0])
            side_test_acc, side_test_auc, side_test_auprc = calculate_metrics(test_outputs[1], test_targets[1], test_loss[1])
            lr_scheduler.step()

            print(f'Epoch {iter}: train acc: {side_train_acc}')
            print(f'Epoch {iter}: val acc: {side_val_acc}')
            print(f'Epoch {iter}: test acc: {side_test_acc}')
            # print(f'Epoch {iter}: train auc: {train_auc}')
            # print(f'Epoch {iter}: val auc: {val_auc}')
            # print(f'Epoch {iter}: test auc: {test_auc}')
            # print(f'Epoch {iter}: train auprc: {train_auprc}')
            # print(f'Epoch {iter}: val auprc: {val_auprc}')
            # print(f'Epoch {iter}: test auprc: {test_auprc}')

            wandb.log({"side_acc": {"train_acc": side_train_acc, "val_acc": side_val_acc, "test_acc": side_test_acc},
                       "duanlie_auc": {"train_auc": duanlie_train_auc, "val_auc": duanlie_val_auc, "test_auc": duanlie_test_auc},
                       "duanlie_auprc": {"train_auprc": duanlie_train_auprc, "val_auprc": duanlie_val_auprc, "test_auprc": duanlie_test_auprc},
                       "side_loss": {"train_loss": train_loss[1] / len(train_targets[1]), "val_loss": val_loss[1] / len(val_targets[1]),
                                "test_loss": test_loss[1] / len(test_targets[1])}
                       }, step=iter)

            if side_test_acc >= best_test_acc:
                # best_epoch_metrics = [round(i, 4) for i in
                #                       [train_acc, val_acc, test_acc, train_auc, val_auc, test_auc, train_auprc,
                #                        val_auprc, test_auprc]]
                best_test_acc = side_test_acc
                test_preds = torch.argmax(test_outputs[1], dim=1)
                fig = plot_matrix(test_targets[1], test_preds, [0, 1, 2],["left", "right", "both"])
                wandb.log({"confusion_matrix": fig}, step=iter)
                # torch.save(models.state_dict(), args["saved_path"] + "/" + version_name + ".pth")

            # if iter % 10 == 0:
            #     save_ckpt(os.path.join(args["ckpt_path"], version_name + ".pth.tar"), models, optimizer, lr_scheduler,
            #               iter, best_test_auc)

        # log_metrics(best_epoch_metrics, args, version_name)
        # with open(args["log_dir"] + "/" + version_name + "/parameters.json", "w+") as f:
        #     json.dump(args, f)
        wandb.finish()


if __name__ == '__main__':
    args = vars(args_parser())
    if args["parameters_path"]:
        with open(args["parameters_path"]) as f:
            args = json.load(f)

    print(args)
    set_seed(args["random_seed"])

    main(args)
