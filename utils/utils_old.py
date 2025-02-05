import os
from tqdm import tqdm

from monai.inferers import sliding_window_inference
from .inferer import sliding_window_inference_with_text

from monai.networks.nets import SwinUNETR, UNet
from models.MyUNet import MyUNet
from models.TextUNet import TextUNet
from models.VAE_UNET import VAE_UNET
from monai.networks.layers import Norm

from monai.data import decollate_batch
import warnings
warnings.filterwarnings("ignore")
from helper import * 
import torch


def validation(args, model, validation_loader, post_label, post_pred, dice_metric):
    model.eval()
    with torch.no_grad():
        for batch in validation_loader:
            val_inputs, val_labels, name, text_embedding = (batch["image"].to(args.device), 
                                                            batch["label"].to(args.device), 
                                                            batch['name'], 
                                                            batch['embed'].to(args.device)
                                                            )
            found = any("Task06_Lung" in text or "Task10_Colon" in text for text in name)
            if found:
                val_labels = (val_labels == 1)*1
            else:
                val_labels = (val_labels == 2)*1

            with torch.cuda.amp.autocast():
                if args.model_name == "text_unet" or "vae_unet":
                    val_outputs = sliding_window_inference_with_text(val_inputs, text_embedding, (96, 96, 96), 4, model)
                else:
                    val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)

            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            
            validation_loader.set_description("Validate (%d / %d Epoch)" % (args.epoch, 10.0))  # noqa: B038
        
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val



def train(args, model, train_loader, loss_function, optimizer, scaler):
    model.train()
    epoch_loss = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)

    for step, batch in enumerate(epoch_iterator):
        x, y, name, text_embedding = (batch["image"].to(args.device), 
                                      batch["label"].to(args.device), 
                                      batch['name'], 
                                      batch['embed'].to(args.device))
        # Creating the binary masks 
        found = any("Task06_Lung" in text or "Task10_Colon" in text for text in name)
        if found:
            y = (y == 1)*1
        else:
            y = (y == 2)*1

        with torch.cuda.amp.autocast():
            if args.model_name == "text_unet" or "vae_unet":
                logit_map = model(x, text_embedding)
            else:
                logit_map = model(x)
            
            loss = loss_function(logit_map, y)
        
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f)" % (
                args.epoch, step, len(train_loader), loss.item())
        )

        epoch_loss += loss.item()
        #torch.cuda.empty_cache()

    print('Epoch=%d: Average_loss=%2.5f' % (args.epoch, epoch_loss/len(epoch_iterator)))
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

    elif args.model_name == 'text_unet':
        model = TextUNet(
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
        raise ValueError("The model specified can not be found. Please select from the list [unet, text_unet, resunet, swi, vae_unet].")

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