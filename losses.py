
import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss3D(nn.Module):
    def __init__(self, window_size=11, reduction='mean', channel=1):
        """
        Initialize the SSIM loss module for 3D data.
        
        Args:
            window_size (int): Size of the Gaussian window.
            reduction (str): Reduction method ('mean' or 'sum').
            channel (int): Number of channels in the input volumes.
        """
        super(SSIMLoss3D, self).__init__()
        self.window_size = window_size
        self.reduction = reduction
        self.channel = channel
        self.window = self.create_gaussian_window(window_size, channel)

    @staticmethod
    def create_gaussian_window(window_size, channel, sigma=1.5):
        gauss_list = []
        for x in range(window_size):
            # Convert index x to a float tensor
            x_t = torch.tensor(x, dtype=torch.float)
            center = torch.tensor(window_size // 2, dtype=torch.float)

            exponent = -(x_t - center)**2 / (2 * sigma**2)
            gauss_val = torch.exp(exponent)
            gauss_list.append(gauss_val)

        gauss = torch.stack(gauss_list)  # shape: (window_size,)
        gauss = gauss / gauss.sum()

        # Then do the outer products as before
        window_1d = gauss.unsqueeze(1)
        window_2d = torch.mm(window_1d, window_1d.t())
        window_3d = torch.einsum('ij,k->ijk', window_2d, gauss)
        
        window = window_3d.unsqueeze(0).unsqueeze(0)
        window = window.repeat(channel, 1, 1, 1, 1)
        return window


    def ssim(self, img1, img2, window, C1=0.01 ** 2, C2=0.03 ** 2):
        """
        Compute SSIM between two 3D volumes.
        
        Args:
            img1 (torch.Tensor): First input volume.
            img2 (torch.Tensor): Second input volume.
            window (torch.Tensor): 3D Gaussian window.
            C1 (float): Constant for stability in luminance calculation.
            C2 (float): Constant for stability in contrast calculation.
            
        Returns:
            torch.Tensor: SSIM value for each volume.
        """
        # Mean values
        mu1 = F.conv3d(img1, window, groups=self.channel, padding=self.window_size // 2)
        mu2 = F.conv3d(img2, window, groups=self.channel, padding=self.window_size // 2)

        # Squared means
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Variance and covariance
        sigma1_sq = F.conv3d(img1 ** 2, window, groups=self.channel, padding=self.window_size // 2) - mu1_sq
        sigma2_sq = F.conv3d(img2 ** 2, window, groups=self.channel, padding=self.window_size // 2) - mu2_sq
        sigma12 = F.conv3d(img1 * img2, window, groups=self.channel, padding=self.window_size // 2) - mu1_mu2

        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return ssim_map

    def forward(self, img1, img2):
        """
        Forward pass for SSIM loss.
        
        Args:
            img1 (torch.Tensor): First input volume.
            img2 (torch.Tensor): Second input volume.
            
        Returns:
            torch.Tensor: SSIM loss value.
        """
        window = self.window.to(img1.device)
        ssim_map = self.ssim(img1, img2, window)

        # Apply reduction
        if self.reduction == 'mean':
            return 1 - ssim_map.mean()
        elif self.reduction == 'sum':
            return 1 - ssim_map.sum()
        else:
            return 1 - ssim_map
        

class LogLikelihoodLossWithCensoring(nn.Module):
    def __init__(self, dist_type):
        """
        Initialize the log-likelihood loss with censoring class.
        
        Args:
            dist_type (str): Distribution type to use for survival prediction (currently only supports 'weibull').
        """
        super(LogLikelihoodLossWithCensoring, self).__init__()
        self.dist_type = dist_type

    def forward(self, inputs, survival_times, risk_status):
        """
        Compute the log-likelihood loss with censoring.
        
        Args:
            inputs (Tensor): Weibull parameters (alpha and lambda) predicted by the model, shape [batch_size, 2].
            survival_times (Tensor): Actual survival times for each sample, shape [batch_size].
            risk_status (Tensor): Binary censoring indicators (1 if event occurred, 0 if censored), shape [batch_size].

        Returns:
            loss (Tensor): Mean log-likelihood loss with censoring.
        """
        survival_times = torch.squeeze(survival_times)
        risk_status = torch.squeeze(risk_status)

        # Extract Weibull parameters from inputs
        alpha = inputs[:, 0]
        lam = inputs[:, 1]

        # Calculate the survival log-probabilities
        survival_term = -torch.pow(survival_times / lam, alpha)

        # Calculate the hazard function where risk_status is 1
        hazard_term = torch.log(alpha / lam) + (alpha - 1) * torch.log(survival_times / lam)

        # Debugging: Check for NaNs or Infs in intermediate calculations
        if torch.any(torch.isnan(survival_term)) or torch.any(torch.isinf(survival_term)):
            print("Warning: NaN or Inf detected in survival_term")
            print("alpha:", alpha)
            print("lam:", lam)
            print("survival_times:", survival_times)
            print("survival_term:", survival_term)

        if torch.any(torch.isnan(hazard_term)) or torch.any(torch.isinf(hazard_term)):
            print("Warning: NaN or Inf detected in hazard_term")
            print("alpha:", alpha)
            print("lam:", lam)
            print("survival_times:", survival_times)
            print("hazard_term:", hazard_term)

        # Apply censoring
        log_likelihood_loss = -torch.sum(hazard_term[risk_status > 0.5]) - torch.sum(survival_term)

        # Mean loss across batch
        loss = log_likelihood_loss / inputs.size(0)

        # Ensure finite loss
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss encountered, check inputs for stability issues.")

        return loss