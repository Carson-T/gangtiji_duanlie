import torch
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--mode', default="J")
    parser.add_argument('--lr', type=list, default=[0.01, 0.01])
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--init_ratio', type=float, default=0.1)
    parser.add_argument('--min_lr_ratio', type=float, default=0.01)
    parser.add_argument('--drop_rate', type=float, default=0)
    parser.add_argument('--drop_path_rate', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resize_h', type=int, default=160)
    parser.add_argument('--resize_w', type=int, default=315)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--is_multiscale', type=int, default=0)
    parser.add_argument('--is_parallel', type=int, default=1)
    parser.add_argument('--use_external', type=int, default=0)
    parser.add_argument('--resume', default="")
    parser.add_argument('--device_ids', type=list, default=[0, 1])
    parser.add_argument('--optim', default="SGD")
    parser.add_argument('--loss_func', default="CEloss")
    parser.add_argument('--init', default="kaiming")
    parser.add_argument('--lr_scheduler', default="StepLR")  #Warm-up-Cosine-Annealing
    parser.add_argument('--backbone', default="resnet50.tv_in1k")  # convnextv2_nano.fcmae_ft_in1k  resnet50.tv_in1k
    parser.add_argument('--model_name',  default="resnet50-J-v2")
    parser.add_argument('--train_csv_path', default="../data/TrainSet/csv/J_train_fold1.csv")
    parser.add_argument('--val_csv_path',  default="../data/TrainSet/csv/J_val_fold1.csv")
    parser.add_argument('--test_path',  default="../data/TestSet")
    parser.add_argument('--external_csv_path', default="../external_data/external_label.csv")
    parser.add_argument('--saved_path', default='../saved_model/resnet')
    parser.add_argument('--ckpt_path', default='../checkpoints/resnet')
    parser.add_argument('--log_dir', default="../logs/resnet")

    args, _ = parser.parse_known_args()
    return args
