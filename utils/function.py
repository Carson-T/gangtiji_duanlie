import torch
import pandas as pd
import os


def xavier(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


def kaiming(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)


def save_ckpt(args, model, optimizer, lr_scheduler, epoch, best_metric):
    state = dict(
        model_state=model.state_dict(),
        optimizer=optimizer.state_dict(),
        lr_scheduler=lr_scheduler.state_dict(),
        epoch=epoch,
        best_test_auc=best_metric,
    )

    torch.save(state, os.path.join(args["ckpt_path"], args["model_name"] + ".pth.tar"))


def log_metrics(best_epoch_metrics, args):
    data = pd.read_csv(args["metrics_log_path"])
    new_data = pd.DataFrame({"model_name": args["model_name"],
                             "train_acc": best_epoch_metrics[0],
                             "valid_acc": best_epoch_metrics[1],
                             "test_acc": best_epoch_metrics[2],
                             "train_auc": best_epoch_metrics[3],
                             "valid_auc": best_epoch_metrics[4],
                             "test_auc": best_epoch_metrics[5],
                             "train_auprc": best_epoch_metrics[6],
                             "valid_auprc": best_epoch_metrics[7],
                             "test_auprc": best_epoch_metrics[8]
                             }, index=[0])
    data = pd.concat([data,new_data])
    data.to_csv(args["metrics_log_path"], index=False)
