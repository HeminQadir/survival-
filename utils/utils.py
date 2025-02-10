import os
from tqdm import tqdm

from monai.inferers import sliding_window_inference
#from .inferer import sliding_window_inference

from monai.networks.nets import SwinUNETR, UNet
#from models.MyUNet import MyUNet
from models.VAE_UNET import VAE_UNET
from models.VAE_GAN import VAE_GAN
from monai.networks.layers import Norm

from monai.data import decollate_batch
import warnings
warnings.filterwarnings("ignore")
from helper import * 
import torch
import os
import random
#import nibabel as nib
import torch.nn.functional as F


def validation(args, model, validation_loader, criterion_bce, phase="train"):
    model.eval()
    epoch_enc_loss = 0
    epoch_dec_loss = 0
    epoch_gan_loss = 0
    
    with torch.no_grad():
        for step, batch in enumerate(validation_loader):
                #model.to(args.device)
                x = batch["image"].to(args.device)
                #print('x.shape0', x.shape[0])
                epoch_enc_loss = 0.0
                epoch_dec_loss = 0.0
                epoch_gan_loss = 0.0
                optimizer_Enc1 = torch.optim.Adam(
                    list(model.encoder.parameters()) + 
                    list(model.conv_mu.parameters()) + 
                    list(model.conv_logvar.parameters()),
                    lr=args.lr
                )
                optimizer_Dec1 = torch.optim.Adam(
                    model.decoder.parameters(), 
                    lr=args.lr
                )
                optimizer_D1 = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr)
                
                real_labels = torch.ones(x.size(0),1).to(args.device)
                
                fake_labels = torch.zeros(x.size(0), 1).to(args.device)
                criterion_bce = criterion_bce.to(args.device)
                criterion_mse = criterion_mse.to(args.device)
                # real_labels = real_labels.to(torch.float16)  # Convert to float16
                # fake_labels = fake_labels.to(torch.float16)
                #print('real_labels',real_labels)
                #with torch.cuda.amp.autocast():
                    
                recon_x, mu, logvar, Dis_x_tilda = model(x)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                Dis_x = model.discriminator(x)
                #print('Dis_X',Dis_x.shape)
                Xp = model.decoder(torch.randn_like(mu))
                Dis_Xp = model.discriminator(Xp.detach())
                #Train Discriminator
                optimizer_D1.zero_grad()
                d_real_loss = criterion_bce(Dis_x, real_labels)
                d_recon_loss = criterion_bce(Dis_x_tilda, fake_labels)
                d_fake_loss = criterion_bce(Dis_Xp, fake_labels)

                Total_Dis_Loss = d_real_loss, d_recon_loss, d_fake_loss
                Total_Dis_Loss.backward(retain_graph =True)
                optimizer_D1.step()

                #Train Generator/Decoder
                optimizer_Dec1.zero_grad()
                rec_loss = criterion_mse(recon_x, x)
                Total_Dec_Loss = 0.001*rec_loss - (d_real_loss, d_recon_loss, d_fake_loss)
                Total_Dec_Loss.backward(retain_graph =True)
                optimizer_Dec1.step()

                #Train Encoder
                Total_Enc_Loss = kl_loss + 0.1*rec_loss
                optimizer_Enc1.zero_grad()
                Total_Enc_Loss.backward()
                optimizer_Enc1.step()

                print("Enc_loss:", Total_Enc_Loss.item())
                print("Dec_loss:", Total_Dec_Loss.item())
                print("loss_GAN:", Total_Dis_Loss.item())

                epoch_enc_loss += Total_Enc_Loss.item()
                epoch_dec_loss += Total_Dec_Loss.item()
                epoch_gan_loss += Total_Dis_Loss.item()
                
        print(f"Epoch {args.epoch}: Avg Encoder Loss={epoch_enc_loss/len(validation_loader):.4f}, Avg Decoder Loss={epoch_dec_loss/len(validation_loader):.4f}, Avg GAN Loss={epoch_gan_loss/len(validation_loader):.4f}")
        return epoch_enc_loss / len(validation_loader), epoch_dec_loss / len(validation_loader), epoch_gan_loss / len(validation_loader), recon_x


def train(args, model, train_loader, optimizer_Enc, optimizer_Dec, optimizer_D,criterion_bce,criterion_mse, scaler):
    
    model.train()

    # for param in model.decoder.parameters():
    #     if param.requires_grad:
    #         print("Decoder params are being updated")
    #     else:
    #         print("Decoder params are frozen!")

    # for param in model.discriminator.parameters():
    #     if param.requires_grad:
    #         print("Discriminator params are being updated")
    #     else:
    #         print("Discriminator params are frozen!")



    for step, batch in enumerate(train_loader):
        #model.to(args.device)
        x = batch["image"].to(args.device)
        #print('x.shape0', x.shape[0])
        epoch_enc_loss = 0.0
        epoch_dec_loss = 0.0
        epoch_gan_loss = 0.0
        optimizer_Enc1 = torch.optim.Adam(
            list(model.encoder.parameters()) + 
            list(model.conv_mu.parameters()) + 
            list(model.conv_logvar.parameters()),
            lr=args.lr
        )
        optimizer_Dec1 = torch.optim.Adam(
            model.decoder.parameters(), 
            lr=args.lr
        )
        optimizer_D1 = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr)
        print('0'*10)
        real_labels = torch.ones(x.size(0),1).to(args.device)
        
        fake_labels = torch.zeros(x.size(0), 1).to(args.device)
        criterion_bce = criterion_bce.to(args.device)
        criterion_mse = criterion_mse.to(args.device)
        # real_labels = real_labels.to(torch.float16)  # Convert to float16
        # fake_labels = fake_labels.to(torch.float16)
        #print('real_labels',real_labels)
        #with torch.cuda.amp.autocast():
        print('1'*10)    
        recon_x, mu, logvar, Dis_x_tilda = model(x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        Dis_x = model.discriminator(x)
        #print('Dis_X',Dis_x.shape)
        Xp = model.decoder(torch.randn_like(mu))
        Dis_Xp = model.discriminator(Xp.detach())
        #Train Discriminator
        optimizer_D1.zero_grad()
        d_real_loss = criterion_bce(Dis_x, real_labels)
        d_recon_loss = criterion_bce(Dis_x_tilda, fake_labels)
        d_fake_loss = criterion_bce(Dis_Xp, fake_labels)
        print('2'*10)
        Total_Dis_Loss = d_real_loss, d_recon_loss, d_fake_loss
        Total_Dis_Loss.backward(retain_graph =True)
        optimizer_D1.step()
        print('3'*10)
        #Train Generator/Decoder
        optimizer_Dec1.zero_grad()
        rec_loss = criterion_mse(recon_x, x)
        Total_Dec_Loss = 0.01*rec_loss - (d_real_loss, d_recon_loss, d_fake_loss)
        print('4'*10)
        Total_Dec_Loss.backward(retain_graph =True)
        print('5'*10)
        optimizer_Dec1.step()
        
        #Train Encoder
        Total_Enc_Loss = kl_loss + 0.01*rec_loss
        optimizer_Enc1.zero_grad()
        Total_Enc_Loss.backward()
        optimizer_Enc1.step()

        print("Enc_loss:", Total_Enc_Loss.item())
        print("Dec_loss:", Total_Dec_Loss.item())
        print("loss_GAN:", Total_Dis_Loss.item())

        epoch_enc_loss += Total_Enc_Loss.item()
        epoch_dec_loss += Total_Dec_Loss.item()
        epoch_gan_loss += Total_Dis_Loss.item()
        
    
    print(f"Epoch {args.epoch}: Avg Encoder Loss={epoch_enc_loss/len(train_loader):.4f}, Avg Decoder Loss={epoch_dec_loss/len(train_loader):.4f}, Avg GAN Loss={epoch_gan_loss/len(train_loader):.4f}")
    return epoch_enc_loss / len(train_loader), epoch_dec_loss / len(train_loader), epoch_gan_loss / len(train_loader)



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
    elif args.model_name == 'vae_gan':
        model = VAE_GAN(
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