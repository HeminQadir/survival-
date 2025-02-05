import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from pycox.evaluation import EvalSurv
import torch

class ci_ibs:
    @staticmethod
    def bern_conf_interval(n, mean, ibs=False):
        """
        Calculate Bernoulli confidence interval using beta distribution.
        
        Args:
            n (int): Total number of observations.
            mean (float): Mean value.
            ibs (bool): Whether it is for integrated Brier score (IBS).

        Returns:
            tuple: (lower confidence bound, mean, upper confidence bound).
        """
        ci_bot, ci_top = proportion_confint(count=mean * n, nobs=n, alpha=0.1, method='beta')
        if mean < 0.5 and not ibs:
            ci_bot_2 = 1 - ci_top
            ci_top = 1 - ci_bot
            ci_bot = ci_bot_2
            mean = 1 - mean

        return np.round(ci_bot, 4), mean, np.round(ci_top, 4)

    @staticmethod
    def obtain_c_index(surv_f, time, censor):
        """
        Evaluate the survival predictions using the concordance index (C-index) and integrated Brier score (IBS).

        Args:
            surv_f (pd.DataFrame): Survival function predictions.
            time (np.ndarray): Array of survival times.
            censor (np.ndarray): Array of censoring indicators.

        Returns:
            tuple: C-index and integrated Brier score (IBS).
        """
        ev = EvalSurv(surv_f, time.flatten(), censor.flatten(), censor_surv='km')
        ci = ev.concordance_td('antolini')

        # Obtain also integrated Brier score
        time_grid = np.linspace(time.min(), time.max(), 100)
        ibs = ev.integrated_brier_score(time_grid)
        return ci, ibs

    def calculate_risk(self, time_train, time_params, censor_val, device=None):
        """
        Calculate risk CDF at specified time points, compute C-index, and IBS with confidence intervals.

        Args:
            time_train (torch.Tensor): Survival times from training data.
            time_params (torch.Tensor): Weibull distribution parameters (alpha, lambda).
            censor_val (torch.Tensor): Censoring indicators.
            device (torch.device, optional): Device for computation.

        Returns:
            tuple: Confidence intervals for C-index and IBS.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Ensure time_train is a NumPy array and get unique survival times
        time_train = time_train.detach().cpu().numpy()
        censor_val = censor_val.detach().cpu().numpy()
        time_params = time_params.detach().cpu().numpy()
        times = np.unique(time_train)

        # Allocate space for predicted risk
        pred_risk = np.zeros((time_params.shape[0], len(times)))

        # Compute predicted risk based on Weibull distribution parameters
        for sample in range(pred_risk.shape[0]):
            alpha = time_params[sample, 0]
            lam = time_params[sample, 1]
            pred_risk[sample, :] = 1 - np.exp(-np.power(times / lam, alpha))

        # Calculate C-index and IBS along with confidence intervals
        ci, ibs = self.obtain_c_index(pd.DataFrame(1 - pred_risk.T, index=times), 
                                      np.asarray(time_train), 
                                      np.asarray(censor_val))
        ci_conf_intervals = self.bern_conf_interval(len(np.asarray(time_train)), ci)
        ibs_conf_intervals = self.bern_conf_interval(len(np.asarray(time_train)), ibs, ibs=True)

        return ci_conf_intervals, ibs_conf_intervals