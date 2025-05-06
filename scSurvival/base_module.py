import torch
import torch.nn as nn
# import torch.nn.functional as F
# import math 

def setup_seed(seed):
    """ Set the random state."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()

        self.V = nn.Linear(input_size, hidden_size, bias=False)
        self.U = nn.Linear(input_size, hidden_size, bias=False)
        self.w = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        atten = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        atten = self.w(atten)
        return atten
    
class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.head_size = hidden_size // n_heads

        self.V = nn.Linear(input_size, hidden_size, bias=False)
        self.U = nn.Linear(input_size, hidden_size, bias=False)
        self.ws = nn.ModuleList([nn.Linear(self.head_size, 1, bias=False) for _ in range(n_heads)])

        # self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        UV = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        # UV = self.dropout(UV)
        #split UV into n_heads
        UV = UV.view(UV.shape[0], -1, self.head_size)

        atts = torch.cat([w(UV[:, i, :]) for i, w in enumerate(self.ws)], dim=1)
        # print('atts:', atts.shape)
        return atts

class SEBlock(nn.Module):
    def __init__(self, feature_dim, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // reduction, feature_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        self.scale = self.fc(x)  # 计算特征权重
        return x * self.scale  # 逐元素相乘，实现特征加权
    
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, use_batch=False,
                 num_batches=None, rec_distribution='G'):
        
        super(VAE, self).__init__()
        setup_seed(42)
        self.use_batch = use_batch
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.rec_distribution = rec_distribution

        if self.use_batch:
            assert num_batches is not None, \
                "num_batches must be provided when use_batch is True"
            self.num_batches = num_batches

            self.encoder_input_dim = input_dim + num_batches
        else:
            self.encoder_input_dim = input_dim

        self.se = SEBlock(input_dim)
        self.encoder_pre = nn.Sequential(
            nn.Linear(self.encoder_input_dim, self.encoder_input_dim // 2),
            # nn.BatchNorm1d(self.encoder_input_dim // 2),
            nn.ReLU(),
            # SEBlock(self.encoder_input_dim // 2),
            nn.Linear(self.encoder_input_dim // 2, hidden_dim),
            nn.ReLU(),
            # SEBlock(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_input_dim = latent_dim + (self.num_batches if self.use_batch else 0)

        self.decoder_pre = nn.Sequential(
            nn.Linear(self.decoder_input_dim, self.decoder_input_dim), 
            nn.ReLU(),
            nn.Linear(self.decoder_input_dim, input_dim // 2),
            nn.ReLU(),
        )
        self.decoder_out = nn.Linear(input_dim // 2, input_dim)

        if rec_distribution == 'ZIG':
            # self.decoder_logvar = nn.Linear(input_dim // 2, input_dim)
            self.decoder_pi = nn.Linear(input_dim // 2, input_dim)
            # self.recon_logvar = nn.Parameter(torch.zeros(1, input_dim), requires_grad=False)
            # random initialization recon_logvar

            vars_gamma = torch.distributions.Gamma(2.0, 2.0).sample((1, input_dim))
            log_vars_gamma = torch.log(vars_gamma + 1e-8)
            self.recon_logvar = nn.Parameter(log_vars_gamma, requires_grad=False)

    def encode(self, x, batch_embed=None):
        x_se = self.se(x)
        if self.use_batch and batch_embed is not None:
            encoder_input = torch.cat([x_se, batch_embed], dim=1)
        else:
            encoder_input = x_se

        h = self.encoder_pre(encoder_input)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, batch_embed=None):
        if self.use_batch and batch_embed is not None:
            decoder_input = torch.cat([z, batch_embed], dim=1)
        else:
            decoder_input = z

        h = self.decoder_pre(decoder_input)
        recon_x = self.decoder_out(h)
 
        if self.rec_distribution == 'ZIG':
            # recon_logvar = self.decoder_logvar(h)
            recon_logvar = self.recon_logvar.expand_as(recon_x)
            recon_logvar = torch.clamp(recon_logvar, min=-10, max=10)

            recon_pi = self.decoder_pi(h)
            recon_pi = torch.sigmoid(recon_pi)
            # recon_pi = torch.clamp(recon_pi, min=0, max=0.7)
            self.recon_pi_avg = torch.mean(recon_pi)
            return recon_x, recon_logvar, recon_pi
        return recon_x
    
    def forward(self, x, batch_embed=None):
        if self.use_batch:
            mu, logvar = self.encode(x, batch_embed)
        else:
            mu, logvar = self.encode(x)
        

        z = self.reparameterize(mu, logvar)
        
        if self.rec_distribution == 'ZIG':
            if self.use_batch:
                recon_x, recon_logvar, recon_pi = self.decode(z, batch_embed)
            else:
                recon_x, recon_logvar, recon_pi = self.decode(z)
            
            return recon_x, mu, logvar, recon_logvar, recon_pi
            # if self.training:
            #     return recon_x, z, logvar, recon_logvar, recon_pi
            # else:
            #     return recon_x, mu, logvar, recon_logvar, recon_pi
        else:
            if self.use_batch:
                recon_x = self.decode(z, batch_embed)
            else:
                recon_x = self.decode(z)
            
            return recon_x, mu, logvar
            # if self.training:
            #     return recon_x, z, logvar
            # else:
            #     return recon_x, mu, logvar

