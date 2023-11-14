import torch
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters_path', default="")

    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--random_seed', type=int, default=2023)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--mode', default="four")
    parser.add_argument('--validation', type=int, default=1)
    parser.add_argument('--fold', type=list, default=[1])
    parser.add_argument('--is_concat', type=int, default=1)
    # parser.add_argument('--combination', type=int, default=1)

    parser.add_argument('--resize_h', type=int, default=224)
    parser.add_argument('--resize_w', type=int, default=224)
    parser.add_argument('--Blur', type=int, default=1)
    parser.add_argument('--OGE', type=int, default=1)
    parser.add_argument('--CLAHE', type=int, default=1)
    parser.add_argument('--Cutout', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sampler', default="")  # WeightedRandomSampler
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--optim', default="SGD")
    parser.add_argument('--lr', type=list, default=[0.001, 0.005])
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--loss_func', default="CEloss")  # LabelSmoothLoss
    parser.add_argument('--use_weighted_loss', type=int, default=1)

    parser.add_argument('--lr_scheduler', default="Warm-up-Cosine-Annealing")  #Warm-up-Cosine-Annealing  StepLR
    parser.add_argument('--init_ratio', type=float, default=0.1)       # Warm-up-Cosine-Annealing parameters
    parser.add_argument('--min_lr_ratio', type=float, default=0.001)    # Warm-up-Cosine-Annealing parameters
    parser.add_argument('--step_size', type=int, default=30)  # StepLR
    parser.add_argument('--gamma', type=float, default=0.1)  # StepLR

    parser.add_argument('--init', default="xavier")   # network initialization methods, kaiming or xavier
    parser.add_argument('--drop_rate', type=float, default=0)
    parser.add_argument('--drop_path_rate', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--is_parallel', type=int, default=1)    # DataParallel or not
    parser.add_argument('--device_ids', type=list, default=[0, 1])    # if is_parallel==1, specify the cuda id

    parser.add_argument('--resume', default="")
    parser.add_argument('--pretrained_path', default="../../pretrained_model/efficientnetv2_rw_s.ra2_in1k.safetensors")
    parser.add_argument('--backbone', default="efficientnetv2_rw_s.ra2_in1k")  # vit_small_patch16_224.augreg_in21k_ft_in1k  efficientnetv2_rw_s.ra2_in1k  convnextv2_nano.fcmae_ft_in1k  resnet50.tv_in1k
    parser.add_argument('--version_name',  default="efficientnetv2_s-3subimg-new_data-four-concat-v1")  # models version
    parser.add_argument('--data_path',  default="../data_3subimg")  # test data path
    parser.add_argument('--saved_path', default='../saved_model/efficientnet')   # models saved path
    parser.add_argument('--ckpt_path', default='../checkpoints/efficientnet')  # checkpoints path for resume
    parser.add_argument('--log_dir', default="..")  # tensorboard/wandb log path
    parser.add_argument('--metrics_log_path', default="../metrics_log.csv")

    args, _ = parser.parse_known_args()
    return args
