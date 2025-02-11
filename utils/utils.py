import os
from tqdm import tqdm

from monai.inferers import sliding_window_inference
#from .inferer import sliding_window_inference

from monai.networks.nets import SwinUNETR, UNet
#from models.MyUNet import MyUNet
from models.VAE_UNET import VAE_UNET
from models.VAE_GAN import Encoder, Decoder,Discriminator
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
#torch.autograd.set_detect_anomaly(True)


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
                print("0"*10)
                optimizer_D1.zero_grad()
                d_real_loss = criterion_bce(Dis_x, real_labels)
                d_recon_loss = criterion_bce(Dis_x_tilda, fake_labels)
                d_fake_loss = criterion_bce(Dis_Xp, fake_labels)
                print("1"*10)
                Total_Dis_Loss = d_real_loss, d_recon_loss, d_fake_loss
                Total_Dis_Loss.backward(retain_graph =True)
                print("Enc_loss:", Total_Enc_Loss.item())
                print("Dec_loss:", Total_Dec_Loss.item())
                print("loss_GAN:", Total_Dis_Loss.item())
                print("2"*10)
                optimizer_D1.step()

                #Train Generator/Decoder
                optimizer_Dec1.zero_grad()
                print("3"*10)
                rec_loss = criterion_mse(recon_x, x)
                Total_Dec_Loss =20*rec_loss - (d_real_loss, d_recon_loss, d_fake_loss)
                Total_Dec_Loss.backward(retain_graph =True)
                optimizer_Dec1.step()
                print("4"*10)
                #Train Encoder
                Total_Enc_Loss = kl_loss + 20*rec_loss
                optimizer_Enc1.zero_grad()
                Total_Enc_Loss.backward()
                optimizer_Enc1.step()

                epoch_enc_loss += Total_Enc_Loss.item()
                epoch_dec_loss += Total_Dec_Loss.item()
                epoch_gan_loss += Total_Dis_Loss.item()
                
        print(f"Epoch {args.epoch}: Avg Encoder Loss={epoch_enc_loss/len(validation_loader):.4f}, Avg Decoder Loss={epoch_dec_loss/len(validation_loader):.4f}, Avg GAN Loss={epoch_gan_loss/len(validation_loader):.4f}")
        return epoch_enc_loss / len(validation_loader), epoch_dec_loss / len(validation_loader), epoch_gan_loss / len(validation_loader), recon_x


# def train(args,encoder,decoder,discriminator, train_loader, optimizer_Enc, optimizer_Dec, optimizer_Dis,criterion_bce,criterion_mse, scaler):
    
#     encoder.train()
#     decoder.train()
#     discriminator.train()
#     encoder.to(args.device)
#     decoder.to(args.device)
#     discriminator.to(args.device)

#     epoch_enc_loss = 0.0
#     epoch_dec_loss = 0.0
#     epoch_dis_loss = 0.0

#     loader_iter = tqdm(train_loader, desc = f"Training VAE-GAN Epoch {args.epoch}", total=len(train_loader))
#     for step, batch in enumerate(loader_iter):
#         x_real = batch["image"].to(args.device)
#         #print('0'*10)
#         real_labels = torch.ones((x_real.size(0),1)).to(args.device)
#         fake_labels = torch.zeros((x_real.size(0),1)).to(args.device)
#         #print('7'*10)
#         criterion_bce = criterion_bce.to(args.device)
#         criterion_mse = criterion_mse.to(args.device)
#         #print('8'*10)
        
           
#         mu, logvar, z = encoder(x_real)
#         #print('9'*10) 
#         x_rec = decoder(z)
#         z_rand = torch.randn_like(mu).to(args.device)
#         #print('10'*10)
#         x_rand = decoder(z_rand)
#         #Train Decoder
#         print('Starting Decoder'*4)
#         optimizer_Dec.zero_grad()
#         d_real_loss = criterion_bce(discriminator(x_real), real_labels)
#         d_recon_loss = criterion_bce(discriminator(x_rec), fake_labels)
#         d_fake_loss = criterion_bce(discriminator(x_rand), fake_labels) 
#         Total_Dec_Loss = (d_real_loss + d_recon_loss+  d_fake_loss)/3 
#         print("Total_Dec_Loss",Total_Dec_Loss)
#         print('1'*10)
#         Total_Dec_Loss.backward(retain_graph =True)
#         print('2'*10)
#         optimizer_Dec.step()
#         print('Ending Decoder'*4)


#         #Train Discriminator
#         print('Starting Discriminator'*4)
#         optimizer_Dis.zero_grad()
#         d_real_loss = criterion_bce(discriminator(x_real), real_labels)
#         output = discriminator(x_rec.detach())
#         d_recon_loss = criterion_bce(output, fake_labels)
#         output = discriminator(x_rand.detach())
#         d_fake_loss = criterion_bce(output, fake_labels) 
#         Total_Dis_Loss = (d_real_loss + d_recon_loss)/3 #+  d_fake_loss
#         gen_img_loss = -Total_Dis_Loss
#         rec_loss = ((x_rec-x_real)**2).mean()
#         Total_Dis_Loss = 20*rec_loss + gen_img_loss
#         print("Total_Dis_Loss",Total_Dis_Loss)
#         print('3'*10)
#         Total_Dis_Loss.backward(retain_graph =True)
#         print('4'*10)
#         optimizer_Dis.step()
#         print('Ending Discriminator'*4)

#         print('Starting Encoder'*4)
#         kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#         Total_Enc_Loss = kl_loss + 10*rec_loss
#         optimizer_Enc.zero_grad()
#         print('5'*10)
#         Total_Enc_Loss.backward()
#         print('Sarting'*10)
#         print('6'*10)
#         optimizer_Enc.step()
#         print('Ending Encoder'*4)



#         #         # Debugging Dec_loss before backward pass
#         # if torch.isnan(Total_Dec_Loss ).any() or torch.isinf(Total_Dec_Loss ).any():
#         #     print(f"NaN or Inf detected in Dec_loss: {Total_Dec_Loss }")
#         #     continue  # Skip this iteration

#         # print(f"Dec_loss.requires_grad: {Total_Dec_Loss .requires_grad}")

#         # # If Dec_loss has no gradient, decoder won't update
#         # for name, param in model.decoder.named_parameters():
#         #     print(f"Warning: Decoder parameter {name} , requires_grad ={param.requires_grad }")

#         print("Enc_loss:", Total_Enc_Loss.item())
#         print("Dec_loss:", Total_Dec_Loss.item())
#         print("Dis_loss:", Total_Dis_Loss.item())

#         epoch_enc_loss += Total_Enc_Loss.item()
#         epoch_dec_loss += Total_Dec_Loss.item()
#         epoch_dis_loss += Total_Dis_Loss.item()
        
    
#     print(f"Epoch {args.epoch}: Avg Encoder Loss={epoch_enc_loss/len(train_loader):.4f}, Avg Decoder Loss={epoch_dec_loss/len(train_loader):.4f}, Avg GAN Loss={epoch_dis_loss/len(train_loader):.4f}")
#     return epoch_enc_loss / len(train_loader), epoch_dec_loss / len(train_loader), epoch_dis_loss / len(train_loader)
# import torch
# from tqdm import tqdm

def train(
    args,
    encoder,
    decoder,
    discriminator,
    train_loader,
    optimizer_Enc,
    optimizer_Dec,
    optimizer_Dis,
    criterion_bce,
    criterion_mse,
    scaler
):
    encoder.train()
    decoder.train()
    discriminator.train()

    encoder.to(args.device)
    decoder.to(args.device)
    discriminator.to(args.device)

    epoch_enc_loss = 0.0
    epoch_dec_loss = 0.0
    epoch_dis_loss = 0.0

    loader_iter = tqdm(train_loader, desc=f"Training VAE-GAN Epoch {args.epoch}", total=len(train_loader))
    for step, batch in enumerate(loader_iter):
        x_real = batch["image"].to(args.device)
        batch_size = x_real.size(0)

        # Setup real/fake labels
        real_labels = torch.ones((batch_size, 1), device=args.device)
        fake_labels = torch.zeros((batch_size, 1), device=args.device)

        # Forward pass: Encoder -> Decoder
        # ---------------------------------------------------
        mu, logvar, z = encoder(x_real)
        x_rec = decoder(z)

        # Also produce random images
        z_rand = torch.randn_like(mu).to(args.device)  # no .detach() needed
        x_rand = decoder(z_rand)


        optimizer_Dis.zero_grad()

        # D sees real => label=1
        d_real = discriminator(x_real)
        loss_real = criterion_bce(d_real, real_labels)

        # D sees recon => label=0
        d_recon = discriminator(x_rec)  # .detach() so gradients won't flow to decoder
        loss_recon = criterion_bce(d_recon, fake_labels)

        # D sees random => label=0
        d_rand = discriminator(x_rand.detach())  # .detach() so gradients won't flow to decoder
        loss_rand = criterion_bce(d_rand, fake_labels)

        dis_loss = (loss_real + loss_recon + loss_rand) / 3.0

        # retain_graph=True because we still have Decoder and Encoder updates next
        dis_loss.backward(retain_graph=True)
        optimizer_Dis.step()
        print("1"*10)
        optimizer_Dec.zero_grad()
        print("2"*10)
        # reconstruction MSE
        rec_loss = criterion_mse(x_rec, x_real)

        # We want the decoder to fool D => label=1 for x_rec
        d_recon_gen = discriminator(x_rec)  # no .detach(), we want grads to flow to decoder
        loss_gen = criterion_bce(d_recon_gen, real_labels)  # fool D => real label

        # Weighted sum: combine reconstruction & adversarial
        dec_loss = 20.0 * rec_loss + 1.0 * loss_gen  # example weighting

        # retain_graph=True because we still have the Encoder
        print("3"*10)
        dec_loss.backward(retain_graph=True)
        print("4"*10)
        optimizer_Dec.step()

  
        optimizer_Enc.zero_grad()

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Usually, we add some reconstruction term for the encoder:
        enc_loss = kl_loss + 200.0 * rec_loss  # Weighted

        # final backward => no retain_graph needed
        print("5"*10)
        enc_loss.backward()
        print("6"*10)
        optimizer_Enc.step()

        epoch_enc_loss += enc_loss.item()
        epoch_dec_loss += dec_loss.item()
        epoch_dis_loss += dis_loss.item()

        loader_iter.set_postfix({
            "Enc_loss": f"{enc_loss.item():.4f}",
            "Dec_loss": f"{dec_loss.item():.4f}",
            "Dis_loss": f"{dis_loss.item():.4f}",
        })

    # End of epoch
    num_batches = len(train_loader)
    avg_enc_loss = epoch_enc_loss / num_batches
    avg_dec_loss = epoch_dec_loss / num_batches
    avg_dis_loss = epoch_dis_loss / num_batches

    print(f"Epoch {args.epoch}: Enc Loss={avg_enc_loss:.4f}, "
          f"Dec Loss={avg_dec_loss:.4f}, Dis Loss={avg_dis_loss:.4f}")

    return avg_enc_loss, avg_dec_loss, avg_dis_loss

import torch
import torch.nn as nn
from tqdm import tqdm

def train(
    args,
    encoder,
    decoder,
    discriminator,
    train_loader,
    optimizer_Enc,
    optimizer_Dec,
    optimizer_Dis,
    criterion_bce,
    criterion_mse,
    scaler,
):
    encoder.train()
    decoder.train()
    discriminator.train()
    encoder.to(args.device)
    decoder.to(args.device)
    discriminator.to(args.device)

    epoch_enc_loss = 0.0
    epoch_dec_loss = 0.0
    epoch_dis_loss = 0.0

    loader_iter = tqdm(
        train_loader, desc=f"Training VAE-GAN Epoch {args.epoch}", total=len(train_loader)
    )
    for step, batch in enumerate(loader_iter):
        # ------------------------------------------------
        # 1) Prepare Inputs & Labels
        # ------------------------------------------------
        x_real = batch["image"].to(args.device)
        batch_size = x_real.size(0)

        real_labels = torch.ones((batch_size, 1), device=args.device)
        fake_labels = torch.zeros((batch_size, 1), device=args.device)

        # Ensure the criteria are on the correct device
        criterion_bce = criterion_bce.to(args.device)
        criterion_mse = criterion_mse.to(args.device)

        # ------------------------------------------------
        # 2) Encoder Forward
        # ------------------------------------------------
        mu, logvar, z = encoder(x_real)  # Encodes real images → (mu, logvar, z)

        # Reconstructed image from the decoder
        x_rec = decoder(z)

        # Random latent → generate fake images
        z_rand = torch.randn_like(mu, device=args.device)
        x_rand = decoder(z_rand)  # purely random generated image

        # ================ TRAIN DECODER FIRST ================
        # The decoder tries to fool the discriminator 
        # as well as produce an image close to real
        optimizer_Dec.zero_grad()

        d_real_loss = criterion_bce(discriminator(x_real), real_labels)
        d_recon_loss = criterion_bce(discriminator(x_rec), fake_labels)  # We want recon to look real → but here it's labeled fake
        d_fake_loss = criterion_bce(discriminator(x_rand), fake_labels)

        # Typically for the decoder, we'd do -d_recon_loss - d_fake_loss if we want them to be real
        # But your code is adding them directly. Let’s keep it consistent but note it’s unusual.
        # We'll do a simple approach: The decoder wants to "fool" the disc, so we do:
        # dec_loss = -(d_recon_loss + d_fake_loss)
        # also add MSE to keep recon close to x_real
        rec_loss = criterion_mse(x_rec, x_real)
        # Scale the reconstruction so it doesn't get overshadowed
        dec_loss = 20.0 * rec_loss - (d_recon_loss + d_fake_loss)
        print("dec_loss", dec_loss)  

        # Check for NaNs
        if torch.isnan(dec_loss).any() or torch.isinf(dec_loss).any():
            print(f"NaN or Inf in dec_loss = {dec_loss.item()}")
            continue
        print("1"*10)    
        dec_loss.backward(retain_graph=True)
        print("2"*10)
        optimizer_Dec.step()

        # ================ TRAIN DISCRIMINATOR ================
        # The discriminator sees real vs. recon vs. random
        optimizer_Dis.zero_grad()

        d_real_loss = criterion_bce(discriminator(x_real), real_labels)
        d_recon_loss = criterion_bce(discriminator(x_rec.detach()), fake_labels)
        d_rand_loss = criterion_bce(discriminator(x_rand.detach()), fake_labels)

        # Summation
        dis_loss = (d_real_loss + d_recon_loss + d_rand_loss) / 3.0

        print("dis_loss", dis_loss)

        # Check for NaNs
        if torch.isnan(dis_loss).any() or torch.isinf(dis_loss).any():
            print(f"NaN or Inf in dis_loss = {dis_loss.item()}")
            continue
        print("3"*10)
        dis_loss.backward(retain_graph=True)
        print("4"*10)
        optimizer_Dis.step()

        # ================ TRAIN ENCODER ================
        # The encoder gets KL + partial recon to ensure a good latent space
        optimizer_Enc.zero_grad()

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        enc_loss = kl_loss + 10.0 * rec_loss  # scale reconstruction

        if torch.isnan(enc_loss).any() or torch.isinf(enc_loss).any():
            print(f"NaN or Inf in enc_loss = {enc_loss.item()}")
            continue
        print("5"*10)
        print("enc_loss:", enc_loss.item())
        enc_loss.backward()
        # try:
        #     enc_loss.backward()
        #     print("enc_loss.backward() completed!")
        # except Exception as e:
        #     print("Exception in enc_loss.backward():", str(e))
        #     # Possibly do "continue" or "break" here.

        print("6"*10)
        optimizer_Enc.step()

        # ================ Logging ================
        epoch_enc_loss += enc_loss.item()
        epoch_dec_loss += dec_loss.item()
        epoch_dis_loss += dis_loss.item()

        loader_iter.set_postfix(
            {
                "Enc_loss": f"{enc_loss.item():.4f}",
                "Dec_loss": f"{dec_loss.item():.4f}",
                "Dis_loss": f"{dis_loss.item():.4f}",
            }
        )

    # End of epoch
    n_batches = len(train_loader)
    avg_enc = epoch_enc_loss / n_batches
    avg_dec = epoch_dec_loss / n_batches
    avg_dis = epoch_dis_loss / n_batches

    print(
        f"Epoch {args.epoch} => "
        f"Enc: {avg_enc:.4f}, Dec: {avg_dec:.4f}, Dis: {avg_dis:.4f}"
    )

    return avg_enc, avg_dec, avg_dis


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
        encoder = Encoder(
                spatial_dims=3,
                in_channels=1,
                out_channels=args.no_class,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2, 1),
                num_res_units=0,
                norm=Norm.BATCH)
        decoder = Decoder(
                spatial_dims=3,
                in_channels=1,
                out_channels=args.no_class,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2, 1),
                num_res_units=0,
                norm=Norm.BATCH)
        discriminator = Discriminator(out_channels=args.no_class)
  

    else: 
        raise ValueError("The model specified can not be found. Please select from the list [unet, my_unet, resunet, swin, vae_unet].")

    # send the model to cuda if available  
    encoder = encoder.to(args.device)

    decoder = decoder.to(args.device)
    discriminator = discriminator.to(args.device)

    args.save_directory = os.path.join(args.save_directory, args.model_name)

    args.path_to_save_results = os.path.join(args.path_to_save_results, args.model_name)

    if args.phase == "train":
        # print the model architecture and number of parameters 
        print(encoder)
        count_parameters(encoder)

    # Define the path to the saved model file
    saved_model_path = os.path.join(args.save_directory, "best_metric_model.pth")

    if args.phase == "train": 
        #Load pre-trained weights
        if args.pretrain is not None:
            encoder.load_params(torch.load(args.pretrain)["state_dict"])
        if args.resume:
            # Check if the path exists
            if os.path.exists(saved_model_path):
                # Load the saved model weights into the model
                encoder.load_state_dict(torch.load(saved_model_path))
                print("The model is restored from a pretrained .pth file")
            else:
                print("Training the model from scratch")
        else:
            print("Training the model from scratch")

    elif args.phase == "test": 
        if args.pretrain is not None:
            encoder.load_state_dict(torch.load(args.pretrain))
        else: 
            raise ValueError("Invalid phase. Provide the path to a trained model to be loaded")
        
    else: 
        raise ValueError("Invalid phase. Please choose 'train' or 'test'.")

    return encoder,decoder,discriminator


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