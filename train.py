import os
from monai.config import print_config
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
from process import process, test
print_config()


def mian(): 

    parser = argparse.ArgumentParser()
    parser.add_argument("--no_class", default=1, type=int,  help="Specify the number of classes, remember the background")   # 2 binary: foreground (tumor) and background
    parser.add_argument("--device", default=2, type=int)
    parser.add_argument("--epoch", default=0)
    parser.add_argument("--fold", default=0, type=int, help="Specify the fold for validation")
    parser.add_argument('--max_epoch', default=2000, type=int, help='Number of training epoches')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-4, help='Weight Decay')
    parser.add_argument('--val_interval', default=1, type=int, help='The interval to start running the model on the validation dataset')
    parser.add_argument('--dataset_json', default='./dataset/dataset_json_files/training_data_nikhil_debug.json', help='data root path')
    parser.add_argument('--save_directory', default='./trained_models', help='path to save the model')
    parser.add_argument('--path_to_save_results', default='./results', help='path to save the results')
    parser.add_argument('--model_name', default='unet', help='backbone [swinunetr or unet or dints or surv_vae_unet or unetpp]')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often, epoch based')
    parser.add_argument('--phase', default='train', help='train or test')

    # Things for dataloader 
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--num_samples', default=4, type=int, help='sample number in each ct')
    # Intensity range 
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    # pixel dimensions
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=2.0, type=float, help='spacing in z direction')
    # spatial size 
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    # do you want to chache? 
    parser.add_argument('--cache_rate', default=1.0, type=float, help='The percentage of cached data in total')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')

    # load from pretrained 
    parser.add_argument('--pretrain', default=None,  #swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
                    help='The path of pretrain model. Eg, ./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt')
    # reload from where it was stoped 
    parser.add_argument('--resume', action="store_true", default=True, help='The path resume from checkpoint')

    args = parser.parse_args()

    # Check for GPU and set it if available
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")


    if args.phase == "train":
        process(args)
    elif args.phase == "test": 
        test(args)
    else:
        raise ValueError("Invalid phase. Please choose 'train' or 'test'.")

if __name__ == "__main__":
    mian()

# python train.py --fold 1 --model_name swin --cache_dataset 
# python train.py --batch_size 5 --device 0 --model_name surv_vae_unet --cache_dataset
# python train.py --batch_size 5 --device 0 --model_name surv_vae_unet --cache_dataset --dataset_json ./dataset/dataset_json_files/training_data_nikhil.json
# python train.py --batch_size 5 --device 0 --model_name my_unet --cache_dataset --dataset_json ./dataset/dataset_json_files/training_data_nikhil_debug.json
# python train.py --device 2 --phase test --pretrain ./trained_models/text_unet/best_metric_model.pth 
# --model_name text_unet --cache_dataset --dataset_json ./dataset/dataset_json_files/training_data_nikhil_debug.json
