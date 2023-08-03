import torch
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--mode', default="J")  # J: jingxi  V: Valsalva
    parser.add_argument('--lr', type=list, default=[0.01, 0.01])
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--init_ratio', type=float, default=0.1)       # Warm-up-Cosine-Annealing parameters
    parser.add_argument('--min_lr_ratio', type=float, default=0.01)    # Warm-up-Cosine-Annealing parameters
    parser.add_argument('--drop_rate', type=float, default=0)
    parser.add_argument('--drop_path_rate', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resize_h', type=int, default=160)
    parser.add_argument('--resize_w', type=int, default=315)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--is_parallel', type=int, default=1)    # DataParallel or not
    parser.add_argument('--use_external', type=int, default=0)  # using external data or not
    parser.add_argument('--resume', default="")
    parser.add_argument('--device_ids', type=list, default=[0, 1])    # if is_parallel==1, specify the cuda id
    parser.add_argument('--optim', default="SGD")
    parser.add_argument('--loss_func', default="CEloss")
    parser.add_argument('--init', default="kaiming")   # network initialization methods, kaiming or xavier 
    parser.add_argument('--lr_scheduler', default="StepLR")  #Warm-up-Cosine-Annealing
    parser.add_argument('--backbone', default="resnet50.tv_in1k")  # pretrained_model  convnextv2_nano.fcmae_ft_in1k  resnet50.tv_in1k
    parser.add_argument('--model_name',  default="resnet50-J-fold1-v1")  # model version
    parser.add_argument('--train_csv_path', default="../data/TrainSet/csv/J_train_fold1.csv")  # train csv path
    parser.add_argument('--val_csv_path',  default="../data/TrainSet/csv/J_val_fold1.csv")   # test csv path
    parser.add_argument('--test_path',  default="../data/TestSet")  # test data path
    parser.add_argument('--saved_path', default='../saved_model/J/resnet')   # model saved path
    parser.add_argument('--ckpt_path', default='../checkpoints/J/resnet')  # checkpoints path for resume
    parser.add_argument('--log_dir', default="../log/J/resnet")  # tensorboard log path

    args, _ = parser.parse_known_args()
    return args
