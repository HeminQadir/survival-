import os
from tqdm import tqdm

from monai.inferers import sliding_window_inference
#from .inferer import sliding_window_inference

from monai.networks.nets import SwinUNETR, UNet
from models.MyUNet import MyUNet
from models.VAE_UNET import VAE_UNET
from monai.networks.layers import Norm

from monai.data import decollate_batch
import warnings
warnings.filterwarnings("ignore")
from helper import * 
import torch
import os
import random
import nibabel as nib
import torch.nn.functional as F


def validation(args, model, validation_loader, MSE_loss, SSIM_loss, phase="train"):
    model.eval()
    epoch_loss = 0
    if phase == "train":
        epoch_iterator = tqdm(validation_loader, desc="Testing on training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    else: 
        epoch_iterator = tqdm(validation_loader, desc="Testing on validation (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            val_inputs, val_labels, event, s_time = (batch["image"].to(args.device), 
                                                     batch["label"].to(args.device), 
                                                     batch['event'].to(args.device), 
                                                     batch['time'].to(args.device))

            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model)
                val_outputs = torch.sigmoid(val_inputs)
                # Update the metrics with your batch of images:
                mse_loss = MSE_loss(val_outputs, val_inputs)
                ssim_loss = SSIM_loss(val_outputs, val_inputs)
                loss = mse_loss + ssim_loss

            epoch_loss += loss.item()

        if phase == "train":
            epoch_iterator.set_description(
                "Epoch=%d: Testing on training (%d / %d Steps) (total_loss=%2.5f)" % (
                    args.epoch, step, len(validation_loader), loss.item())
            )
        else:
            epoch_iterator.set_description(
                "Epoch=%d: Testing on validation (%d / %d Steps) (total_loss=%2.5f)" % (
                    args.epoch, step, len(validation_loader), loss.item())
            )
            
    if  phase == "train":
        print('Epoch=%d: Average__loss_on_training=%2.5f' % (args.epoch, epoch_loss/len(epoch_iterator)))
    else:
        print('Epoch=%d: Average__loss_on_validation=%2.5f' % (args.epoch, epoch_loss/len(epoch_iterator)))
    return epoch_loss/len(epoch_iterator), val_outputs


def train(args, model, train_loader, MSE_loss, SSIM_loss, SURV_loss, optimizer, scaler):
    model.train()
    epoch_loss = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)

    for step, batch in enumerate(epoch_iterator):
        x, y, event, s_time = (batch["image"].to(args.device), 
                               batch["label"].to(args.device), 
                               batch['event'].to(args.device), 
                               batch['time'].to(args.device))

        with torch.cuda.amp.autocast():
            logit_map, mu, logvar, weibull_params = model(x)
            logit_map = torch.sigmoid(logit_map)
            mse_loss = MSE_loss(logit_map, x)
            ssim_loss = SSIM_loss(logit_map, x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            surv_loss = SURV_loss(weibull_params, s_time, event)
            loss = mse_loss + ssim_loss + kl_loss + surv_loss
            
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (total_loss=%2.5f)" % (
                args.epoch, step, len(train_loader), loss.item())
        )

        epoch_loss += loss.item()
        #torch.cuda.empty_cache()

    print('Epoch=%d: Average_train_loss=%2.5f' % (args.epoch, epoch_loss/len(epoch_iterator)))
    return epoch_loss/len(epoch_iterator)


def model_setup(args):
    # loading the model
    if args.model_name == 'swin':
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=args.no_class, 
            feature_size=48,
            #use_checkpoint=True,
        )
    
    elif args.model_name == 'unet':
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=args.no_class,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=0,
            norm=Norm.BATCH,
        ) 

    elif args.model_name == 'my_unet':
        model = MyUNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=args.no_class,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2, 1),
                num_res_units=0,
                norm=Norm.BATCH
        )
    elif args.model_name == 'vae_unet':
        model = VAE_UNET(
                spatial_dims=3,
                in_channels=1,
                out_channels=args.no_class,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2, 1),
                num_res_units=0,
                norm=Norm.BATCH
        )

    else: 
        raise ValueError("The model specified can not be found. Please select from the list [unet, my_unet, resunet, swin, vae_unet].")

    # send the model to cuda if available  
    model = model.to(args.device)

    args.save_directory = os.path.join(args.save_directory, args.model_name)

    args.path_to_save_results = os.path.join(args.path_to_save_results, args.model_name)

    if args.phase == "train":
        # print the model architecture and number of parameters 
        print(model)
        count_parameters(model)

    # Define the path to the saved model file
    saved_model_path = os.path.join(args.save_directory, "best_metric_model.pth")

    if args.phase == "train": 
        #Load pre-trained weights
        if args.pretrain is not None:
            model.load_params(torch.load(args.pretrain)["state_dict"])
        if args.resume:
            # Check if the path exists
            if os.path.exists(saved_model_path):
                # Load the saved model weights into the model
                model.load_state_dict(torch.load(saved_model_path))
                print("The model is restored from a pretrained .pth file")
            else:
                print("Training the model from scratch")
        else:
            print("Training the model from scratch")

    elif args.phase == "test": 
        if args.pretrain is not None:
            model.load_state_dict(torch.load(args.pretrain))
        else: 
            raise ValueError("Invalid phase. Provide the path to a trained model to be loaded")
        
    else: 
        raise ValueError("Invalid phase. Please choose 'train' or 'test'.")

    return model


def validation_old(args, model, validation_loader, post_label, post_pred, dice_metric):
    model.eval()
    with torch.no_grad():
        for batch in validation_loader:
            val_inputs, val_labels, event, s_time = (batch["image"].to(args.device), 
                                batch["label"].to(args.device), 
                                batch['event'].to(args.device), 
                                batch['time'].to(args.device))
    
            print("I am in validation "*5)
            print(val_inputs.shape)
        
            with torch.cuda.amp.autocast():
                print(val_inputs.shape)
                val_outputs = sliding_window_inference(val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model)

            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            
            validation_loader.set_description("Validate (%d / %d Epoch)" % (args.epoch, 10.0))  # noqa: B038
        
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val