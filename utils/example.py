def train(args, model, train_loader, optimizer_Enc, optimizer_Dec, optimizer_D,criterion_bce, scaler):
    
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
        model.to(args.device)
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
        #print('Dis_X',Dis_x)
        loss_GAN = (criterion_bce(Dis_x, real_labels) + criterion_bce(Dis_x_tilda, fake_labels) + criterion_bce(Dis_Xp, fake_labels)) / 3
        print('2'*10)
        LDislLike = criterion_bce(Dis_x, real_labels)  # BCE loss for real images
        print('3'*10)
        Enc_loss = kl_loss + LDislLike
        Dec_loss = 0.01 * LDislLike - loss_GAN
        print('4'*10) 

        print("Enc_loss:", Enc_loss.item())
        print("Dec_loss:", Dec_loss.item())
        print("loss_GAN:", loss_GAN.item())
        
        # Check for NaNs or Infs in the loss
        if torch.isnan(Enc_loss).any() or torch.isinf(Enc_loss).any():
            print(f"NaN or Inf detected in Enc_loss: {Enc_loss}")
            continue

        if torch.isnan(Dec_loss).any() or torch.isinf(Dec_loss).any():
            print(f"NaN or Inf detected in Dec_loss: {Dec_loss}")
            continue

        if torch.isnan(loss_GAN).any() or torch.isinf(loss_GAN).any():
            print(f"NaN or Inf detected in loss_GAN: {loss_GAN}")
            continue
        # Debugging: Check if decoder and discriminator contribute to loss 
        print(f"Dec_loss.requires_grad: {Dec_loss.requires_grad}") 
        print(f"loss_GAN.requires_grad: {loss_GAN.requires_grad}")

        optimizer_Enc1.zero_grad()
        Enc_loss.backward()
        optimizer_Enc1.step()
        print('5'*10)

        optimizer_Dec1.zero_grad()
        Dec_loss.backward()
        optimizer_Dec1.step()
        print('6'*10)
        optimizer_D1.zero_grad()
        loss_GAN.backward()
        optimizer_D1.step()
        print('7'*10)
        epoch_enc_loss += Enc_loss.item()
        epoch_dec_loss += Dec_loss.item()
        epoch_gan_loss += loss_GAN.item()
        print('8'*10)  


    
    print(f"Epoch {args.epoch}: Avg Encoder Loss={epoch_enc_loss/len(train_loader):.4f}, Avg Decoder Loss={epoch_dec_loss/len(train_loader):.4f}, Avg GAN Loss={epoch_gan_loss/len(train_loader):.4f}")
    return epoch_enc_loss / len(train_loader), epoch_dec_loss / len(train_loader), epoch_gan_loss / len(train_loader)
