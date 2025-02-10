import os

from tqdm import tqdm
from dataset.data_split_fold import datafold_read, datafold_read_inference
from monai.losses import DiceCELoss, DiceLoss
from losses import SSIMLoss3D
from torch.nn import MSELoss 
from monai.transforms import AsDiscrete

from monai.metrics import DiceMetric
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
from tensorboardX import SummaryWriter
from dataset.dataloader import get_loader, get_loader_inference 
from utils.utils import *
from helper import * 
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torchvision 

def process(args):

    # Setup the model 
    model, D = model_setup(args) 

    # check if this directory is available, if not make it 
    make_dir(args.save_directory)

    # Split the dataset into train and validation subsets 
    train_files, val_files = datafold_read(args) 

    # get the data loader an apply augmenations  
    train_loader, val_loader, subset_loader = get_loader(train_files, val_files, args)

    torch.backends.cudnn.benchmark = True
    MSE_loss = MSELoss(reduction='mean')
    SSIM_loss = SSIMLoss3D(window_size=11, reduction='mean', channel=1)

    #loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) #, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    
    optimizerD = torch.optim.Adam(D.parameters(), lr=args.lr)
    scalerD = torch.cuda.amp.GradScaler()

    loss_val_best = 100.0
    
    writer = SummaryWriter(log_dir='out/' + args.model_name)
    print('Writing Tensorboard logs to ', 'out/' + args.model_name)

    while args.epoch < args.max_epoch:

        try: 
            # Traininig 
            ave_loss, kl_loss, dis_loss = train(args, model, D, train_loader, MSE_loss, SSIM_loss, optimizer, optimizerD, scaler, scalerD)
            writer.add_scalar('train_loss', ave_loss, args.epoch)
            writer.add_scalar('train_kl_loss', kl_loss, args.epoch)
            writer.add_scalar('train_dis_loss', dis_loss, args.epoch)
            
            if (args.epoch % args.store_num == 0 and args.epoch != 0):
                    torch.save(model.state_dict(), os.path.join(args.save_directory, "model_epoch_"+str(args.epoch)+"_.pth"))
            
            # Validation 
            if (args.epoch % args.val_interval == 0):
                print("starting the validation phase")
                val_inputs, loss_val, val_outputs = validation(args, model, val_loader, MSE_loss, SSIM_loss, phase="validation") 
                train_inputs, loss_train, train_outputs = validation(args, model, subset_loader, MSE_loss, SSIM_loss, phase="train") 

                if loss_val < loss_val_best:
                    loss_val_best = loss_val

                    torch.save(model.state_dict(), os.path.join(args.save_directory, "best_metric_model.pth"))
                    print(
                        "Model Was Saved ! Current Best Avg. Rec Loss: {} Current Avg. Rec Loss: {}".format(loss_val_best, loss_val)
                    )
                else:
                    print(
                        "Model Was Not Saved ! Current Best Avg. Rec Loss: {} Current Avg. Rec Loss: {}".format(
                            loss_val_best, loss_val
                        )
                    )
                
                writer.add_scalar("Validation/Average Accuracy:", scalar_value=torch.tensor(loss_val), global_step=args.epoch+1)
                writer.add_scalar("Train/Average Accuracy:", scalar_value=torch.tensor(loss_train), global_step=args.epoch+1)

                # for j in range(0, args.batch):
                #     slice_index = val_outputs.shape[-1] // 2
                #     writer.add_image('rec image', torch.tensor(val_outputs[j, :, :, slice_index]), global_step=args.epoch+1)
                slice_index = val_outputs.shape[-1] // 2
                writer.add_image('recon image', torch.tensor(val_outputs[0, :, :, slice_index]), global_step=args.epoch+1)
                writer.add_image('input image', torch.tensor(val_inputs[0, :, :, slice_index]),  global_step=args.epoch+1)

        except: 
            torch.save(model.state_dict(), os.path.join(args.save_directory, "last_epoch_model.pth"))

        
        args.epoch += 1


    torch.save(model.state_dict(), os.path.join(args.save_directory, "last_epoch_model.pth"))
    print(f"train completed, best_metric: {loss_val_best:.4f}" f"at iteration: {args.epoch}")



def test(args):

    # Setup the model and load a pretrained model if provided 
    model = model_setup(args) 

    # check if this directory is available, if not make it 
    make_dir(args.path_to_save_results)

    # Split the dataset into train and validation subsets 
    #val_files = datafold_read_inference(args) 
    train_files, val_files = datafold_read(args) 

    # get the data loader an apply augmenations  
    val_loader = get_loader_inference(val_files, args)

    torch.backends.cudnn.benchmark = True

    post_label = AsDiscrete(to_onehot=args.no_class)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.no_class)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Epoch) (dice=X.X)", dynamic_ncols=True)
    print("starting the validation phase")
    dice_val = validation(args, model, epoch_iterator_val, post_label, post_pred, dice_metric) 

    print(dice_val)