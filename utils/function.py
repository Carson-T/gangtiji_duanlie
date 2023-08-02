import torch
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
        best_performance=best_metric,
    )

    torch.save(state, os.path.join(args["ckpt_path"], args["model_name"]+".pth.tar"))