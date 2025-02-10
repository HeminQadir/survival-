import os
from tqdm import tqdm

from monai.inferers import sliding_window_inference
#from .inferer import sliding_window_inference

from monai.networks.nets import SwinUNETR, UNet
from models.MyUNet import MyUNet
from models.VAE_UNET import VAE_UNET
from models.SURV_VAE_UNET import SURV_VAE_UNET
from models.Discriminator import Discriminator
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
                #val_outputs, z, log_var, mu, weibull_params = sliding_window_inference(val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model)
                val_outputs, z, log_var, mu, weibull_params = model(val_inputs) 
                val_outputs = torch.sigmoid(val_outputs)
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
        print('Epoch=%d: Average_loss_on_training=%2.5f' % (args.epoch, epoch_loss/len(epoch_iterator)))
    else:
        print('Epoch=%d: Average_loss_on_validation=%2.5f' % (args.epoch, epoch_loss/len(epoch_iterator)))
    return val_inputs, epoch_loss/len(epoch_iterator), val_outputs


def train(args, model, D, train_loader, MSE_loss, SSIM_loss, optimizer, optimizerD, scaler, scalerD):
    model.train()
    D.train()
    epoch_loss = 0
    epoch_kl = 0
    epoch_dis = 0
    
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)

    #These things are for the D
    real_label = torch.ones(args.batch_size * args.num_samples, 1).to(args.device)  
    fake_label = torch.zeros(args.batch_size * args.num_samples, 1).to(args.device) 
    criterion = torch.nn.BCELoss()

    for step, batch in enumerate(epoch_iterator):
        x, y, event, s_time = (batch["image"].to(args.device), 
                               batch["label"].to(args.device), 
                               batch['event'].to(args.device), 
                               batch['time'].to(args.device))

        # This the G 
        logit_map, z, log_var, mu, weibull_params = model(x)
        x_rec = torch.sigmoid(logit_map)
        mse_loss = MSE_loss(x_rec, x)
        ssim_loss = SSIM_loss(x_rec, x)
        kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        kl_weight = 0.00025
        
    
        z = torch.randn(args.batch_size * args.num_samples, 1024, requires_grad=False).to(args.device) 
        z = model.fc3(z)
        x_rand = model.decoder(z)
        
        
        ###############################################
        # Train D 
        ###############################################
        #Real loss
        # optimizerD.zero_grad()
        
        d_real_loss = criterion(D(x), real_label)
        d_fake_loss = criterion(D(x_rand), fake_label)
        d_recon_loss = criterion(D(x_rec), fake_label)
        
        dis_loss = d_real_loss + d_fake_loss + d_recon_loss
       
        # dis_loss.backward(retain_graph=True)
        
        # optimizerD.step()
        
        
        scalerD.scale(dis_loss).backward(retain_graph=True)
        scalerD.unscale_(optimizerD)
        scalerD.step(optimizerD)
        scalerD.update()
        optimizerD.zero_grad()
        

        ###############################################
        # Train G
        ###############################################
        # optimizer.zero_grad()
        
        output_x = D(x)
        d_real_loss = criterion(output_x, real_label)
        output_rec = D(x_rec)
        d_recon_loss = criterion(output_rec, fake_label)
        output_z = D(x_rand)
        d_fake_loss = criterion(output_z, fake_label)

        d_img_loss = d_real_loss + d_recon_loss+ d_fake_loss
        gen_img_loss = -d_img_loss
        gamma = 20
        err_dec = gamma * (mse_loss + ssim_loss) + gen_img_loss + (kl_weight * kl_divergence)
        
        # err_dec.backward(retain_graph=True)
        # optimizer.step()
        
    
        scaler.scale(err_dec).backward()
        epoch_loss += err_dec.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        


        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (total_loss=%2.5f) (Dis loss= %2.5f)" % (
                args.epoch, step, len(train_loader), err_dec.item(), dis_loss.item())
        )


        epoch_loss += err_dec.item()

        epoch_kl += kl_divergence.item()
        
        epoch_dis+=dis_loss.item()

        #torch.cuda.empty_cache()

    #print('Epoch=%d: Average_train_loss=%2.5f' % (args.epoch, epoch_loss/len(epoch_iterator)))
    #print('Epoch=%d: Average_train_kl_loss=%2.5f' % (args.epoch, kl_divergence/len(epoch_iterator)))
    
    return epoch_loss/len(epoch_iterator), epoch_kl/len(epoch_iterator), epoch_dis/len(epoch_iterator)


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
    elif args.model_name == 'surv_vae_unet':
        model = SURV_VAE_UNET(
                spatial_dims=3,
                in_channels=1,
                out_channels=args.no_class,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2, 1),
                num_res_units=0,
                norm=Norm.BATCH
        )

    else: 
        raise ValueError("The model specified can not be found. Please select from the list [unet, my_unet, surv_vae_unet, resunet, swin, vae_unet].")

    # send the model to cuda if available  
    model = model.to(args.device)

    D = Discriminator(
                spatial_dims=3,
                in_channels=1,
                out_channels=args.no_class,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2, 1),
                num_res_units=0,
                norm=Norm.BATCH
                )
    D = D.to(args.device)

    args.save_directory = os.path.join(args.save_directory, args.model_name)

    args.path_to_save_results = os.path.join(args.path_to_save_results, args.model_name)

    if args.phase == "train":
        # print the model architecture and number of parameters 
        print(model)
        count_parameters(model)
        print("this is the discriminator")
        print(D)
        count_parameters(D)

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

    return model, D


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