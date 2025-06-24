import torch
import torch.nn as nn
import torch.nn.functional as F

from .scsurvival_module import scSurvivalCellModel, HazrdModel, ProjectorModel
from .loss_func import cox_loss_func, c_index, conditional_cindex
# from random import shuffle, sample
from tqdm import tqdm
import numpy as np

import random
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast
from contextlib import nullcontext
from .utils import GPUDataLoader
from sklearn.model_selection import train_test_split
from collections import deque

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import pandas as pd

def setup_seed(seed):
    """ Set the random state."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

class scSurvival(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_heads=None, extract_feature=True, use_batch=False, num_batches=None, beta=0.1, tau=0.2, rec_distribution='ZIG', dropout=0.5, covariate_size=None):
        super(scSurvival, self).__init__()
        setup_seed(42)
        self.model = None
        self.hidden_size = hidden_size
        self.use_batch = use_batch
        self.num_batches = num_batches
        
        self.ae_criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cell_model = scSurvivalCellModel(input_size, hidden_size, 
                                              num_heads=num_heads, extract_feature=extract_feature, use_batch=use_batch, num_batches=num_batches,
                                              beta=beta, tau=tau, rec_distribution=rec_distribution)
        self.hazard_model = HazrdModel(hidden_size, dropout=dropout, covariate_size=covariate_size)
        
        self.cell_model.to(self.device)
        self.hazard_model.to(self.device)

        self.num_heads = num_heads
        if num_heads is not None:
            self.projecters = nn.ModuleList([ProjectorModel(hidden_size, num_heads) for _ in range(num_heads)])
            self.projecters.to(self.device)

        self.extract_feature = extract_feature

    def pretrain_epoch(self, all_dataloader, feature_weights, gamma_beta_weight,  scaler, amp_context, use_amp, optimizer, lamda=1.0, num_iter=None, train=True):
        ae_loss = 0.0
        count = 0
        for data_batch in all_dataloader:
            optimizer.zero_grad()
            with amp_context:
                if self.use_batch:
                    instance_batch, batch_labels = data_batch
                    instance_batch = instance_batch if instance_batch.device == self.device else instance_batch.to(self.device)
                    batch_labels = batch_labels if batch_labels.device == self.device else batch_labels.to(self.device)
                    _, _, ae_loss_batch = self.cell_model(instance_batch, batch_labels, feature_weight=feature_weights,gamma_beta_weight=gamma_beta_weight,
                    disable_attn=True)
                else:
                    instance_batch = data_batch[0] if data_batch[0].device == self.device else data_batch[0].to(self.device)
                    # print('instance_batch:', instance_batch.shape)
                    _, _, ae_loss_batch = self.cell_model(instance_batch, feature_weight=feature_weights,  
                    gamma_beta_weight=gamma_beta_weight,
                    disable_attn=True)

                ae_loss_batch /= instance_batch.shape[0]

            if train:
                loss_ae = lamda * ae_loss_batch        
                if use_amp:
                    scaler.scale(loss_ae).backward()
                    torch.nn.utils.clip_grad_norm_(self.cell_model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_ae.backward()
                    optimizer.step()

            ae_loss += ae_loss_batch

            count += 1
            if num_iter is not None and count >= num_iter:
                break

        ae_loss /= count

        if train:
            return ae_loss.detach()
        else:
            return ae_loss

    def fit(self, 
            xs, 
            y_time, y_event, 
            covariates_encoded=None,
            batch_lists=None, 
            validate=False,
            validate_ratio=0.3,
            validate_metric='ccindex', 
            feature_weights=None, 
            epochs=500, pretrain_epochs=200, lr=0.001, 
            pretrain_batch_size=None,
            instance_batch_size=None, 
            lambdas=(0.01, 1.0), 
            entropy_threshold=0.7, 
            weight_decay=0.01, 
            patience=100, 
            temperature=1, 
            finetue_lr_factor=0.1, 
            gamma_beta_weight=(0.1, 0.0), 
            once_load_to_gpu=True,
            use_amp=False,
            fitnetune_strategy='alternating_lightly',
            sample_balance=False, 
            **kwargs):
        
        '''
        Parameters
        ----------
        xs: list of expression matrices, each matrix represents a single-cell RNA-seq data.
        y_event: tensor, event indicator.
        covariates_encoded: tensor, covariates encoded by dummy variables.
        batch_lists: list of batch indices, each one represents the batch labels of a single-cell RNA-seq data.
        validate: bool, whether to perform validation during training. If True, will split 20% of the data for validation.
        validate_ratio: float, the ratio of data to use for validation when `validate` is True. Default is 0.2 (20%).
        validate_metric: str, 'ccindex' or 'cindex', the metric to use for validation when `validate` is True. Default is 'ccindex'.
        feature_weights: 1D array, feature weights for the reconstruction loss in the autoencoder.
        epochs: int, number of epochs for training.
        pretrain_epochs: int, number of epochs for pretraining the autoencoder.
        lr: float, learning rate.
        pretrain_batch_size: int, batch size for pretraining the autoencoder. 
            If None, it will be set to instance_batch_size.
        instance_batch_size: int, batch size for training the model. 
            If None, it will be set to the maximum length of the input data.
        lambdas: tuple, (lambda_ae, lambda_entropy), the weight for the reconstruction loss and the entropy regularization term for cell attention.
        entropy_threshold: float, the threshold for the entropy regularization term. 
            The less the entropy, the cell attention is more focused on the most important cells. If one wants to keep more high attention cells,
            the entropy should be larger.
        weight_decay: float, weight decay for the optimizer.
        patience: int, number of epochs for early stopping. 
        temperature: float, temperature for the softmax function in the cell attention.
        finetue_lr_factor: float, the learning rate factor for the encoder during finetuning. 
            The learning rate for the encoder will be set to lr * finetue_lr_factor.
        gamma_beta_weight: tuple, (gamma_weight, beta_weight), the weight for the gamma and beta prior in the autoencoder.
        once_load_to_gpu: bool, whether to load the data to GPU once or not.
        use_amp: bool, whether to use automatic mixed precision or not.
        fitnetune_strategy: jointly, alternating, alternating_lightly
            1. jointly: jointly train autoencoder and cox model
            2. alternating: alternating train autoencoder and cox model
            3. alternating_lightly: alternating train autoencoder and cox model, but the autoencoder is trained one batch at each epoch.
        sample_balance: bool, whether to sample balance the data or not.
            If True, the data will be sampled to have the same number of cells for each input data.
       ''' 
        # setup_seed(42)
        if validate:
            print(f'Validation mode is enabled, will split {int(validate_ratio * 100)}% of the data for validation.')
            train_idx, test_idx = train_test_split(range(len(xs)), test_size=validate_ratio, random_state=42, stratify=y_event, shuffle=True)
            xs_train = [xs[i] for i in train_idx]
            xs_test = [xs[i] for i in test_idx]
            y_time_train = y_time[train_idx]
            y_event_train = y_event[train_idx]
            y_time_test = y_time[test_idx]
            y_event_test = y_event[test_idx]

            xs, y_time, y_event = xs_train, y_time_train, y_event_train
            if self.use_batch and batch_lists is not None:
                batch_lists_train = [batch_lists[i] for i in train_idx]
                batch_lists_test = [batch_lists[i] for i in test_idx]
                batch_lists = batch_lists_train  
            
            if covariates_encoded is not None:
                covariates_encoded_train = covariates_encoded.iloc[train_idx]
                covariates_encoded_test = covariates_encoded.iloc[test_idx]
                covariates_encoded = covariates_encoded_train

        if self.device.type == 'cpu':
            once_load_to_gpu = False
        if once_load_to_gpu:
            xs = [torch.tensor(x, dtype=torch.float32, device=self.device) for x in xs]
        else:
            xs = [torch.tensor(x, dtype=torch.float32) for x in xs]

        if self.use_batch:
            assert batch_lists is not None, 'batch_lists should be provided when use_batch is True'
            if once_load_to_gpu:
                batch_lists = [torch.tensor(batch, dtype=torch.long, device=self.device) for batch in batch_lists]
            else:
                batch_lists = [torch.tensor(batch, dtype=torch.long) for batch in batch_lists]
            batch_lists = [F.one_hot(batch, num_classes=self.num_batches) for batch in batch_lists]

        y_event = torch.tensor(y_event, dtype=torch.float32)
        y_event = y_event.view(-1, 1)
        y_time = torch.tensor(y_time, dtype=torch.float32)
        sorted_idx = torch.argsort(y_time.view(-1), dim=0, descending=True)
        xs = list(map(xs.__getitem__, sorted_idx))
        if self.use_batch:
            batch_lists = list(map(batch_lists.__getitem__, sorted_idx))
        
        if covariates_encoded is not None:
            covariates_encoded_features = covariates_encoded.columns
            covariates_encoded = torch.tensor(covariates_encoded.values, dtype=torch.float32)
            covariates_encoded = covariates_encoded[sorted_idx]
            covariates_encoded = covariates_encoded.to(self.device)

        y_time = y_time[sorted_idx]
        y_event = y_event[sorted_idx]

        y_event = y_event.to(self.device)
        y_time = y_time.to(self.device)
        
        atten_params = list(self.cell_model.attn.parameters())
        atten_params_set = set(atten_params)
        cellembd_params = [p for p in self.cell_model.parameters() if p not in atten_params_set]

        if self.num_heads is not None:
            atten_params += list(self.projecters.parameters())

        hazard_params = list(self.hazard_model.parameters())
        cellembd_lr = lr * finetue_lr_factor if pretrain_epochs > 0 else lr
        param_groups = [
            {'params': cellembd_params, 'lr': cellembd_lr, 'weight_decay': weight_decay},
            {'params': atten_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': hazard_params, 'lr': lr, 'weight_decay': weight_decay}
        ]

        # self.optimizer = torch.optim.Adam(list(self.cell_model.parameters()) + list(self.hazard_model.parameters()), lr=lr)
        if weight_decay > 0:
            self.optimizer = torch.optim.AdamW(param_groups)
        else:
            self.optimizer = torch.optim.Adam(param_groups)
        self.optimizer_cell = torch.optim.Adam(cellembd_params, lr=lr)
        
        max_len = max([x.shape[0] for x in xs])
        if instance_batch_size is None:   
            instance_batch_size = max_len

        if use_amp:
            scaler = GradScaler()
            amp_context = autocast(device_type=self.device.type)
        else:
            amp_context = nullcontext()
            scaler = None

        x_all = torch.cat(xs, dim=0)
        if sample_balance:
            sample_size = int(x_all.shape[0] / len(xs))
            instance_batch_size = sample_size

        if pretrain_batch_size is None:
            pretrain_batch_size = instance_batch_size

        def get_dataloader(self):
            if self.use_batch:
                batch_all = torch.cat(batch_lists, dim=0)
                
                if once_load_to_gpu:
                    all_dataloader = GPUDataLoader((x_all, batch_all), batch_size=pretrain_batch_size, shuffle=True)
                else:
                    all_dataset = TensorDataset(x_all, batch_all)
                    all_dataloader = DataLoader(all_dataset, batch_size=pretrain_batch_size, shuffle=True)
            else:
                if once_load_to_gpu:
                    all_dataloader = GPUDataLoader(x_all, batch_size=pretrain_batch_size, shuffle=True)
                else:
                    all_dataset = TensorDataset(x_all)
                    all_dataloader = DataLoader(all_dataset, batch_size=pretrain_batch_size, shuffle=True)
            return all_dataloader

        if pretrain_epochs > 0:
            progress = tqdm(range(pretrain_epochs), desc='Pretraining', leave=True)

            all_dataloader = get_dataloader(self)
            for _ in progress:
                self.train()
                ae_loss = self.pretrain_epoch(all_dataloader, feature_weights, gamma_beta_weight, scaler, amp_context, use_amp, self.optimizer_cell)
                progress.set_postfix(ae_loss=ae_loss.item())    

        best_metric = float('inf')
        recent_metrics = deque(maxlen=5)
        best_model = None 
        patience_count = 0
        if pretrain_epochs > 0:
            progress = tqdm(range(epochs), desc='Finetuning', leave=True)
            for module in self.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.eval()
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False
        else:
            progress = tqdm(range(epochs), desc='Training', leave=True)

        all_dataloader = get_dataloader(self)
        for _ in progress:
            self.train()
            # phase 1 
            if fitnetune_strategy == 'jointly':
                train = False
                num_iter = None
            elif fitnetune_strategy == 'alternating':
                train = True
                num_iter = None
            elif fitnetune_strategy == 'alternating_lightly':
                train = True
                num_iter = 1
            ae_loss = self.pretrain_epoch(all_dataloader, feature_weights, gamma_beta_weight, scaler, amp_context, use_amp, self.optimizer, lamda=lambdas[0], num_iter=num_iter, train=train)  # jointly train autoencoder

            # phase 2
            self.optimizer.zero_grad()
            atten_entropy = 0.0
            hazards = []
            h_alls = []

            if len(xs) > y_event.shape[0]:
                # means that there are some patients without survival information
                xs_to_cox = xs[:y_event.shape[0]]
                batch_lists_to_cox = batch_lists[:y_event.shape[0]]
            else:
                xs_to_cox = xs
                batch_lists_to_cox = batch_lists

            if sample_balance:
                xs_to_cox_ = []
                batch_lists_to_cox_ = []

                for i, each in enumerate(xs_to_cox):
                    if sample_size > each.shape[0]:
                        sample_indices = random.choices(range(each.shape[0]), k=sample_size)
                    else:
                        sample_indices = random.sample(range(each.shape[0]), sample_size)

                    if once_load_to_gpu:
                        sample_indices = torch.tensor(sample_indices, dtype=torch.long, device=self.device)

                    xs_to_cox_.append(each[sample_indices])
                    if self.use_batch:
                        batch_lists_to_cox_.append(batch_lists_to_cox[i][sample_indices])

                xs_to_cox = xs_to_cox_
                batch_lists_to_cox = batch_lists_to_cox_

            # optimize whole model
            with amp_context:
                for i, x in enumerate(xs_to_cox):
                    hs, attens = [], []
                    if self.use_batch:
                        batch_labels = batch_lists_to_cox[i]

                    for j in range(0, len(x), instance_batch_size):
                        to_idx = min(j+instance_batch_size, x.shape[0])
                        instance_batch = x[j:to_idx]
                        instance_batch = instance_batch if instance_batch.device == self.device else instance_batch.to(self.device)
                        if self.use_batch:
                            batch_labels_batch = batch_labels[j:to_idx]
                            batch_labels_batch = batch_labels_batch if batch_labels_batch.device == self.device else batch_labels_batch.to(self.device)
                            h, a, _ = self.cell_model(instance_batch, batch_labels_batch, feature_weight=feature_weights, 
                            gamma_beta_weight=gamma_beta_weight, disable_decoder=True)
                        else:
                            h, a, _ = self.cell_model(instance_batch, feature_weight=feature_weights, gamma_beta_weight=gamma_beta_weight, disable_decoder=True)

                        hs.append(h)
                        attens.append(a)
                    
                    hs = torch.cat(hs, dim=0)
                    attens = torch.cat(attens, dim=0)
                    attens = F.softmax(attens / temperature, dim=0)


                    atten_entropy += (-torch.sum(attens * torch.log(attens + 1e-7), dim=0) / np.log(attens.shape[0])).mean()

                    if self.num_heads is not None:
                        h_all = [torch.sum(p(hs) * attens[:, i].view(-1, 1), dim=0) for i, p in enumerate(self.projecters)]
                        h_all = torch.cat(h_all, dim=0)
                    else:
                        h_all = torch.sum(hs * attens, dim=0)

                    h_alls.append(h_all)

                h_alls = torch.cat(h_alls, dim=0).view(-1, h_alls[0].shape[0])
                hazards = self.hazard_model(h_alls, covariates=covariates_encoded) 

                atten_entropy /= len(xs)
                cox_loss = cox_loss_func(hazards, y_event)

                if lambdas[1] == 0.0:
                    loss = cox_loss
                else:
                    loss = cox_loss + lambdas[1] * torch.relu(atten_entropy - entropy_threshold)
            
            loss = loss + lambdas[0] * ae_loss
            if use_amp:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # loss = loss.detach() + lambdas[0] * ae_loss
            
            
            if validate:
                # validate on test set
                self.eval()
                with torch.no_grad():
                    hazards_test = self.predict_samples(
                        xs=xs_test, 
                        instance_batch_size=instance_batch_size, 
                        batch_lists=batch_lists_test if self.use_batch else None,
                        covariates_encoded=covariates_encoded_test if covariates_encoded is not None else None
                        )
                    if validate_metric == 'cindex':
                        # directly use cindex on the test set hazards
                        y_time_test = torch.tensor(y_time_test, dtype=torch.float32, device=self.device).view(-1)
                        y_event_test = torch.tensor(y_event_test, dtype=torch.float32, device=self.device).view(-1)
                        cindex_test = c_index(
                            hazards_test.view(-1),
                            y_time_test,
                            y_event_test
                        ).item()

                        cindex_test_rev = 1 - cindex_test
                        recent_metrics.append(cindex_test_rev )  # store 
                        avg_recent = np.mean(recent_metrics) if len(recent_metrics) > 0 else float('inf')

                        if self.extract_feature:
                            progress.set_postfix(loss=loss.item(), cox_loss=cox_loss.item(), ae_loss=ae_loss.item(), atten_entropy=atten_entropy.item(), cindex_val=1-avg_recent)
                        else:
                            progress.set_postfix(loss=loss.item(), cox_loss=cox_loss.item(), atten_entropy=atten_entropy.item(), cindex_val=1-avg_recent)
                            
                    else:
                        all_risk_scores = np.hstack([
                            hazards.cpu().numpy().reshape(-1),
                            hazards_test.cpu().numpy().reshape(-1) 
                        ])

                        all_y_event = np.hstack([
                            y_event.cpu().numpy().reshape(-1),
                            y_event_test.reshape(-1)
                        ])

                        all_y_time = np.hstack([
                            y_time.cpu().numpy().reshape(-1),
                            y_time_test.reshape(-1)
                        ])

                        subset_ids = np.arange(y_time.shape[0])
                        ccindex_test = conditional_cindex(
                            all_y_time, 
                            all_y_event, 
                            all_risk_scores,
                            subset=subset_ids
                        )
                    
                
                        ccindex_test_rev = 1 - ccindex_test
                        recent_metrics.append(ccindex_test_rev )  # store 
                        avg_recent = np.mean(recent_metrics) if len(recent_metrics) > 0 else float('inf')

                        if self.extract_feature:
                            progress.set_postfix(loss=loss.item(), cox_loss=cox_loss.item(), ae_loss=ae_loss.item(), atten_entropy=atten_entropy.item(), ccindex_val=1-avg_recent)
                        else:
                            progress.set_postfix(loss=loss.item(), cox_loss=cox_loss.item(), atten_entropy=atten_entropy.item(), ccindex_val=1-avg_recent)
                
                if avg_recent < 0.45:
                    if best_metric > avg_recent:
                        best_metric = avg_recent
                        patience_count = 0
                        best_model = {
                            'cell_model': deepcopy(self.cell_model.state_dict()),
                            'hazard_model': deepcopy(self.hazard_model.state_dict()),
                        }
                        if self.num_heads is not None:
                            best_model['projecters'] = deepcopy(self.projecters.state_dict())
                    else:
                        # early stopping
                        patience_count += 1
                        if patience_count >= patience:
                            print(f"Early stopping with best validation {validate_metric}: {(1-best_metric):.4f}.")
                            if best_model is not None:
                                self.cell_model.load_state_dict(best_model['cell_model'])
                                self.hazard_model.load_state_dict(best_model['hazard_model'])
                                if self.num_heads is not None:
                                    self.projecters.load_state_dict(best_model['projecters'])
                            break

            else:
                # recent_metrics.append(loss.item())
                # avg_recent = np.mean(recent_metrics) if len(recent_metrics) > 0 else float('inf')

                # if self.extract_feature:
                #     progress.set_postfix(loss=avg_recent, cox_loss=cox_loss.item(), ae_loss=ae_loss.item(), atten_entropy=atten_entropy.item())
                # else:
                #     progress.set_postfix(loss=avg_recent, cox_loss=cox_loss.item(), atten_entropy=atten_entropy.item())

                # if best_metric > avg_recent:
                #     best_metric = avg_recent
                #     patience_count = 0
                #     # best_model = {
                #     #     'cell_model': deepcopy(self.cell_model.state_dict()),
                #     #     'hazard_model': deepcopy(self.hazard_model.state_dict()),
                #     # }
                #     # if self.num_heads is not None:
                #     #     best_model['projecters'] = deepcopy(self.projecters.state_dict())
                # else:
                #     patience_count += 1
                #     if patience_count >= patience:
                #         # print(f"Early stopping with best loss: {best_metric:.4f}.")
                #         # if best_model is not None:
                #         #     self.cell_model.load_state_dict(best_model['cell_model'])
                #         #     self.hazard_model.load_state_dict(best_model['hazard_model'])
                #         #     if self.num_heads is not None:
                #         #         self.projecters.load_state_dict(best_model['projecters'])
                #         break

                if self.extract_feature:
                    progress.set_postfix(loss=loss.item(), cox_loss=cox_loss.item(), ae_loss=ae_loss.item(), atten_entropy=atten_entropy.item())
                else:
                    progress.set_postfix(loss=loss.item(), cox_loss=cox_loss.item(), atten_entropy=atten_entropy.item())

                if best_metric > loss.item():
                    best_metric = loss.item()
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= patience:
                        break
            
        # for hessian to do wald test
        if covariates_encoded is not None:

            hazards_base = self.hazard_model(h_alls).detach()
            covariates_encoded_features = list(covariates_encoded_features)

            from lifelines import CoxPHFitter
            cph = CoxPHFitter(penalizer=0.1)

            df = pd.DataFrame(
                data=covariates_encoded.cpu().numpy(),
                columns=covariates_encoded_features
            )
            df['scSurvival_hazard'] = hazards_base.cpu().numpy()
            df['y_event'] = y_event.cpu().numpy()
            df['y_time'] = y_time.cpu().numpy()
            df['y_event'] = df['y_event'].astype(int) 
            df['y_time'] = df['y_time'].astype(float)

            cov_weight = self.hazard_model.covariate_hazard.weight.detach().view(-1).cpu().numpy()

            cov_weight = np.append(cov_weight, 1.0)
            cph.fit(df, 
                    duration_col='y_time', 
                    event_col='y_event', 
                    initial_point=cov_weight,
                    show_progress=False)  
            # cph.print_summary()
            self.cph = cph
            self.covariate_coef = cph.summary.copy()

    def predict_cells(self, x, batch_labels=None,
                      batch_size=10000):
        # x should be a tensor representing a single-cell RNA-seq data
        x = torch.tensor(x, dtype=torch.float32)
        if self.use_batch:
            batch_labels = torch.tensor(batch_labels, dtype=torch.long)
            batch_labels = F.one_hot(batch_labels, num_classes=self.num_batches)

        if batch_labels is not None:
            dataset = TensorDataset(x, batch_labels)
        else:
            dataset = TensorDataset(x)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.eval()

        h_alls = []
        a_alls = []
        cell_hazard_alls = []
        cell_hazard_weighted_alls = []
        for i, data_batch in enumerate(dataloader):
            x = data_batch[0].to(self.device)
            if self.use_batch:
                batch_labels = data_batch[1].to(self.device)
            else:
                batch_labels = None
            with torch.no_grad():
                # x = torch.tensor(x, dtype=torch.float32).to(self.device)
                # if batch_labels is not None:
                #     batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(self.device)
                #     batch_labels = F.one_hot(batch_labels, num_classes=self.num_batches)

                h, a, _ = self.cell_model(x, batch_embed=batch_labels, disable_decoder=True)

                a = torch.sigmoid(a)
                a_sigmoid = deepcopy(a)
                a = a.max(dim=1)[0] if self.num_heads is not None else a

                if self.num_heads is not None:
                    h_all_cells = [p(h) for i, p in enumerate(self.projecters)]
                    h_all_cells = torch.cat(h_all_cells, dim=1)
                    cell_hazards = self.hazard_model(h_all_cells)

                    h_all_cells_weighted = [p(h) * a_sigmoid[:, i].view(-1, 1) for i, p in enumerate(self.projecters)]
                    h_all_cells_weighted = torch.cat(h_all_cells_weighted, dim=1)
                    cell_hazards_weighted = self.hazard_model(h_all_cells_weighted)
                else:
                    cell_hazards = self.hazard_model(h)
                    cell_hazards_weighted = self.hazard_model(h * a_sigmoid)
                h_alls.append(h)
                a_alls.append(a)
                cell_hazard_alls.append(cell_hazards)
                cell_hazard_weighted_alls.append(cell_hazards_weighted)
        
        h = torch.cat(h_alls, dim=0)
        a = torch.cat(a_alls, dim=0)
        cell_hazards = torch.cat(cell_hazard_alls, dim=0)
        cell_hazards_weighted = torch.cat(cell_hazard_weighted_alls, dim=0)
        return h, a, cell_hazards, cell_hazards_weighted
    
    def predict_samples(self, xs, instance_batch_size=None, batch_lists=None, covariates_encoded=None):
        # xs should be a list of tensors representing single-cell RNA-seq data
        max_len = max([x.shape[0] for x in xs])
        if instance_batch_size is None:
            instance_batch_size = max_len

        if batch_lists is not None:
            batch_lists = [torch.tensor(batch, dtype=torch.long, device=self.device) for batch in batch_lists]
            batch_lists = [F.one_hot(batch, num_classes=self.num_batches) for batch in batch_lists]
        
        if covariates_encoded is not None:
            covariates_encoded = torch.tensor(covariates_encoded.values, dtype=torch.float32)
            covariates_encoded = covariates_encoded.to(self.device)

        with torch.no_grad():
            self.eval()
            hazards = []
            h_alls = []
            for i, x in enumerate(xs):
                hs, attens = [], []
                x = torch.tensor(x, dtype=torch.float32)
                for j in range(0, len(x), instance_batch_size):
                    to_idx = min(j+instance_batch_size, x.shape[0])
                    instance_batch = x[j:to_idx].to(self.device)
                    if batch_lists is not None:
                        batch_labels = batch_lists[i][j:to_idx].to(self.device)
                        h, a, _ = self.cell_model(instance_batch, batch_embed=batch_labels, disable_decoder=True)
                    else:
                        h, a, _ = self.cell_model(instance_batch, disable_decoder=True)
                    hs.append(h)
                    attens.append(a)

                hs = torch.cat(hs, dim=0)
                attens = torch.cat(attens, dim=0)
                attens = F.softmax(attens, dim=0)

                if self.num_heads is not None:
                    h_all = [torch.sum(p(hs) * attens[:, i].view(-1, 1), dim=0) for i, p in enumerate(self.projecters)]
                    h_all = torch.cat(h_all, dim=0)
                else:
                    h_all = torch.sum(hs * attens, dim=0)

                # hazard = self.hazard_model(h_all)
                # hazards.append(hazard)
                h_alls.append(h_all)

            # hazards = torch.cat(hazards, dim=0).view(-1)
            h_alls = torch.cat(h_alls, dim=0).view(-1, h_alls[0].shape[0])
            hazards = self.hazard_model(h_alls, covariates=covariates_encoded) if covariates_encoded is not None else self.hazard_model(h_alls)
        
        return hazards

class CovPreprocessor:
    def __init__(self):
        self.continuous_cols = None
        self.categorical_cols = None
        self.pipeline = None
        self.feature_names = None

    def fit(self, df):
        # Identify categorical columns first
        self.categorical_cols = df.select_dtypes(include=['object', 'category', 'bool', 'string']).columns.tolist()
        self.continuous_cols = list(set(df.columns) - set(self.categorical_cols))

        # Build the preprocessing pipeline
        transformers = [
            ('num', StandardScaler(), self.continuous_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), self.categorical_cols)
        ]
        self.pipeline = ColumnTransformer(transformers)
        self.pipeline.fit(df)

        # Store feature names for transformed DataFrame
        cat_features = self.pipeline.named_transformers_['cat'].get_feature_names_out(self.categorical_cols)
        self.feature_names = self.continuous_cols + cat_features.tolist()

    def transform(self, df):
        if self.pipeline is None:
            raise ValueError("You must call fit() before transform().")

        # Check that all required columns are present
        expected_cols = set(self.continuous_cols + self.categorical_cols)
        missing_cols = expected_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"The following required columns are missing from input data: {sorted(missing_cols)}")

        transformed = self.pipeline.transform(df)
        return pd.DataFrame(transformed, columns=self.feature_names, index=df.index)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def add_survival_columns(self, X, duration, event):
        """
        Add duration and event columns to the covariate matrix.
        """
        X = X.copy()
        X['duration'] = duration
        X['event'] = event.astype(int)
        return X

    def save(self, path_prefix='cov_preprocessor'):
        joblib.dump(self.pipeline, f'{path_prefix}_pipeline.pkl')
        joblib.dump((self.continuous_cols, self.categorical_cols, self.feature_names), f'{path_prefix}_meta.pkl')

    def load(self, path_prefix='cov_preprocessor'):
        self.pipeline = joblib.load(f'{path_prefix}_pipeline.pkl')
        self.continuous_cols, self.categorical_cols, self.feature_names = joblib.load(f'{path_prefix}_meta.pkl')
