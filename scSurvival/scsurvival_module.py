import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_module import *
from .loss_func import *

class scSurvivalCellModelAE(nn.Module):
    def __init__(self, input_size, hidden_size=None, num_heads=None, extract_feature=True, **kwargs):
        super(scSurvivalCellModelAE, self).__init__()
        self.extract_feature = extract_feature
        if extract_feature:
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(), 
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size ),
                nn.ReLU(),
                nn.Linear(hidden_size, input_size),
            )
        else:
            self.feature_extractor = nn.Identity()
            hidden_size = input_size

        if num_heads is None:
            self.attn = Attention(hidden_size, hidden_size)
        else:
            self.attn = MultiHeadAttention(hidden_size, hidden_size, num_heads)

    def forward(self, x, feature_weight=None, **kwargs):
        h = self.feature_extractor(x)
        a = self.attn(h)
        # a = F.softmax(a, dim=1)
        if self.extract_feature:
            decoded_x = self.decoder(h)
            if feature_weight is None:
                ae_loss = F.mse_loss(decoded_x, x, reduction='sum')
            else:
                if feature_weight.sum() != feature_weight.shape[0]:
                    feature_weight = feature_weight / feature_weight.mean()
                ae_loss = F.mse_loss(decoded_x, x, reduction='none')
                ae_loss = (ae_loss * feature_weight).sum()
        else:
            ae_loss = 0
        return h, a, ae_loss


class scSurvivalCellModelVAE(nn.Module):
    def __init__(self, input_size, hidden_size=None, num_heads=None, 
                 extract_feature=True, 
                 beta=0.1, tau=0.2, 
                 use_batch=False, num_batches=None,
                 rec_distribution='G',  
                 **kwargs):
        '''
        rec_distribution: G: Guassian, ZIG: Zero Inflation Guassian. G+L: Gaussian + Laplace. 
        '''
        super(scSurvivalCellModelVAE, self).__init__()
        self.extract_feature = extract_feature
        self.beta = beta
        self.tau = tau
        self.use_batch = use_batch
        self.rec_distribution = rec_distribution

        if extract_feature:
            self.feature_extractor = VAE(input_size, hidden_size, hidden_size, use_batch=use_batch, num_batches=num_batches, rec_distribution=rec_distribution)
        else:
            self.feature_extractor = nn.Identity()
            hidden_size = input_size

        if num_heads is None:
            self.attn = Attention(hidden_size, hidden_size)
        else:
            self.attn = MultiHeadAttention(hidden_size, hidden_size, num_heads)

        self.input_ln = nn.LayerNorm(input_size)
        # self.input_ln = nn.BatchNorm1d(input_size)
        # self.input_ln = nn.Identity()
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma_prior = torch.distributions.Gamma(
                            torch.tensor([2.0], device=device),
                            torch.tensor([2.0], device=device)
                            )
        self.beta_prior = torch.distributions.Beta(
                            torch.tensor([5.0], device=device),
                            torch.tensor([2.0], device=device)
                            )

    def forward(self, x, batch_embed=None, feature_weight=None, gamma_beta_weight=(0.1, 0.1), disable_attn=False, disable_decoder=False):
            
        if self.extract_feature:
            if self.rec_distribution == 'ZIG':
                if disable_decoder:
                    if self.use_batch:
                        h, _ = self.feature_extractor.encode(self.input_ln(x), batch_embed)
                    else:
                        h, _ = self.feature_extractor.encode(self.input_ln(x))

                    ae_loss = torch.tensor(0.0)
                else:
                    if self.use_batch:
                        decoded_x, h, logvar, recon_logvar, recon_pi = self.feature_extractor(self.input_ln(x), batch_embed)
                    else:
                        decoded_x, h, logvar, recon_logvar, recon_pi = self.feature_extractor(self.input_ln(x))
                    ae_loss = zero_inflated_gaussian_loss(
                        x, decoded_x, recon_logvar, recon_pi, 
                        reduction='none', 
                        gamma_beta_weight=gamma_beta_weight,
                        gamma_dist=self.gamma_prior, 
                        beta_dist=self.beta_prior
                        )
            else:
                if self.use_batch:
                    decoded_x, h, logvar = self.feature_extractor(self.input_ln(x), batch_embed)
                else:
                    decoded_x, h, logvar = self.feature_extractor(self.input_ln(x))
                if self.rec_distribution == 'G+L':
                    ae_loss = mse_mae_loss(x, decoded_x, reduction='none')
                else:
                    ae_loss = F.mse_loss(decoded_x, x, reduction='none') 

            if disable_decoder:
                kld_loss = 0
            else:
                if feature_weight is None:
                    ae_loss = ae_loss.sum()
                else:
                    if feature_weight.sum() != feature_weight.shape[0]:
                        feature_weight = feature_weight / feature_weight.mean()
                    ae_loss = (ae_loss * feature_weight).sum()

                kld_loss_per_item = -0.5 * (1 + logvar - h.pow(2) - logvar.exp())
                kld_loss_per_item = torch.clamp(kld_loss_per_item-self.tau, min=0)
                kld_loss = torch.sum(kld_loss_per_item)

            ae_loss = ae_loss + self.beta * kld_loss
        else:
            h = self.feature_extractor(x)
            ae_loss = 0

        if disable_attn:
            a = None
        else:
            a = self.attn(h)

        return h, a, ae_loss
    
scSurvivalCellModel = scSurvivalCellModelVAE
# scSurvivalCellModel = scSurvivalCellModelAE

class HazrdModel(nn.Module):
    def __init__(self, hidden_size, dropout=0.5, covariate_size=None):
        super(HazrdModel, self).__init__() 

        self.hazard = nn.Sequential(
            # SEBlock(hidden_size),
            # nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            # nn.ReLU(),
            # nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

        # self.hazard = nn.Sequential(
        #     nn.Linear(hidden_size, 1)
        # )

        if covariate_size is not None:
            self.covariate_hazard = nn.Linear(covariate_size, 1, bias=False)
            # self.raw_gamma = nn.Parameter(torch.zeros(1))
            # self.gamma_range = 0.1

    def forward(self, h, covariates=None):
        hazard = self.hazard(h)
        hazard = torch.clamp(hazard, min=-10, max=10)
        if covariates is not None:
            # hazard = hazard + self.covariate_hazard(covariates)
            # gamma = 1 + torch.tanh(self.raw_gamma) * self.gamma_range
            # hazard = self.covariate_hazard(covariates) + hazard * gamma

            hazard = self.covariate_hazard(covariates) + hazard
        return hazard 
    

class ProjectorModel(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(ProjectorModel, self).__init__()

        self.projction = nn.Sequential(
            # nn.Dropout(0.9),
            nn.Linear(hidden_size, hidden_size // num_heads),
            # nn.BatchNorm1d(hidden_size // num_heads),
            # nn.Dropout(0.9),
        )

    def forward(self, x):
        x = self.projction(x)
        return x