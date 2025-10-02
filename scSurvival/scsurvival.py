from .scsurvival_core import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scanpy.external.pp import harmony_integrate
import scanpy as sc
import pandas as pd
import os 

def robust_disp_inv(exp, eps=1e-6):
    m = np.mean(exp, axis=0)
    std = np.std(exp, axis=0)
    disp_inv = m / (std + eps)
    return disp_inv

def calc_gene_weights(exp, alpha=0.2, eps=1e-6):
    vals = np.mean(exp, axis=0)
    # vals = np.std(exp, axis=0)
    # vals = robust_disp_inv(exp, eps=eps)
    
    vals_median = np.median(vals)
    w = (vals + eps) / (vals_median + eps)
    w = np.power(w, -alpha)
    w = w / np.mean(w)
    return w

def scSurvivalRun(adata, sample_column, surv, covariates=None, batch_key=None, feature_flavor='AE', beta=0.1, tau=0.2, hidden_size=128, num_heads=8, rec_likelihood='ZIG', do_scale_ae=False, gene_weight_alpha=0.2, model_save_dir=None, model_load_dir=None, dropout=0.5, predict_nMC=64, feature_key=None, **kwargs):
    '''
    Parameters
    ----------
    adata: AnnData object. X should be logh-normalized expression data.
    sample_column: sample column name in adata.obs. 
        The samples without survival info can also be inculded. They can be integrated for batch correction, and be predicted cell hazards and sample hazards. When the samples which need to be predicted have significant batch effect with the training samples, it is recommended to include them in the training set, i.e., adata.
    surv: survival data, Two-column dDataFrame with index as sample_colum, and columns as ['time ', 'status']
    covariates: covariates for survival analysis, such as age, gender, etc.
        The covariates should be a DataFrame with index as sample_column, and columns as covariates.
    batch_key(optional): batch key in adata.obs
    feature_flavor: 'PCA' , 'AE' or 'Custom'. 
        PCA: use PCA as feature. If batch_key is not None, use harmony corrected PCA. If pca key is not in adata.obsm, PCA is performed automatically first.
        AE: Jointly train model with an autoencoder to extract features. Only support batch_key=None. Next steps may be added to support batch_key using the VAE like scVI. 
        Custom: use custom features in adata.obsm[feature_key]. feature_key is required.
    beta: beta parameter for KLD loss in variational autoencoder.
    tau: tau freebits parameter for KLD loss in variational autoencoder.
    hidden_size: hidden size of the model.
    num_heads: number of attention heads.
    rec_likelihood: reconstruction likelihood for autoencoder. Currently support 'ZIG': Zero-inflated Gaussian and 'G': Gaussian.   
    do_scale_ae: whether to scale the adata.
    gene_weight_alpha: alpha parameter for gene weights. See calc_gene_weights() for the definition.
    model_save_dir: directory to save the model.
    model_load_dir: directory to load the model.
    dropout: dropout rate for the hazard model.
    predict_nMC: number of Monte Carlo samples for prediction. Only used when `sample_balance` is True in training.
    feature_key: key in adata.obsm for custom features. Required when feature_flavor is 'Custom'.
    kwargs: other parameters for scSurvival.fit().
    

    Returns
    -------
    adata: AnnData object with added hazard and attention in adata.obs.
    surv: survival data with added patient_hazards.
    model: trained scSurvival model.

    See Also
    --------
    See scSurvival.fit() for more parameters.

    '''
    if feature_flavor == 'PCA' and 'X_pca' in adata.obsm.keys():
        pass
    else:
        if adata.X.min() < 0 or adata.X.max() > 20:
            raise ValueError('adata.X should be log-normalized expression data. Please preprocess it first.')

    patients = adata.obs[sample_column].unique()
    patients_with_surv = [p for p in patients if p in surv.index]
    patients_wo_surv = [p for p in patients if p not in surv.index]
    patients = patients_with_surv + patients_wo_surv #patients without survival data must be at the end.

    surv = surv.loc[patients_with_surv, :]
    y_time = surv['time'].values
    y_event = surv['status'].values

    # initialize batch lists
    batch_lists = None

    if covariates is not None:
        cov_psr = CovPreprocessor()
        covariates_encoded = cov_psr.fit_transform(covariates.loc[patients_with_surv, :])
        covariate_size = covariates_encoded.shape[1]
    else:
        cov_psr = None
        covariate_size = None
        covariates_encoded = None

    if feature_flavor == 'Custom':
        assert feature_key is not None, 'feature_key is required for Custom feature_flavor.'
        assert feature_key in adata.obsm.keys(), 'feature_key %s not in adata.obsm' % feature_key
        xs = [adata[adata.obs[sample_column] == each].obsm[feature_key] for each in patients]
        input_size = xs[0].shape[1] 
        model = scSurvival(input_size, 
                           hidden_size=input_size, 
                           num_heads=num_heads, 
                           extract_feature=False, 
                           dropout=dropout, 
                           covariate_size=covariate_size)
        
        if 'entropy_threshold' not in kwargs:
            kwargs['entropy_threshold'] = 0.5
        if model_load_dir is not None:
            model.load_state_dict(torch.load(os.path.join(model_load_dir, 'model.pt'), map_location=model.device, weight_only=True))
            print('Model loaded from %s' % model_load_dir)
        model.fit(xs, 
                  y_time, 
                  y_event, 
                  covariates_encoded=covariates_encoded,
                #   epochs=500, 
                #   instance_batch_size=3000, 
                #   lambdas=(0.0, 1.0),
                #   entropy_threshold=0.5, 
                  **kwargs)
        _, a, cell_hazards, cell_hazards_weighted = model.predict_cells(adata.obsm[feature_key])

        model.feature_key = feature_key
        batch_lists = None

    elif feature_flavor == 'PCA':
        pca_key = None
        if batch_key is None:
            if 'X_pca' not in adata.obsm.keys():
                print('Run PCA...')
                sc.pp.scale(adata, max_value=10)
                sc.tl.pca(adata, n_comps=50, svd_solver='arpack', zero_center=False)
            pca_key = 'X_pca'
        else:
            if 'X_pca_harmony' not in adata.obsm.keys():
                print('Run Harmony...')
                if 'X_pca' not in adata.obsm.keys():
                    sc.pp.scale(adata, max_value=10)
                    sc.tl.pca(adata, n_comps=50, svd_solver='arpack', zero_center=False)
                harmony_integrate(adata, key=batch_key, max_iter_harmony=20, verbose=False) 
            pca_key = 'X_pca_harmony'

        xs = [adata[adata.obs[sample_column] == each].obsm[pca_key] for each in patients]
        input_size = xs[0].shape[1]

        model = scSurvival(input_size, 
                           hidden_size=input_size, 
                           num_heads=5, 
                           extract_feature=False, dropout=dropout, covariate_size=covariate_size)
        
        if 'entropy_threshold' not in kwargs:
            kwargs['entropy_threshold'] = 0.5

        if model_load_dir is not None:
            model.load_state_dict(torch.load(os.path.join(model_load_dir, 'model.pt'), map_location=model.device, weight_only=True))
            print('Model loaded from %s' % model_load_dir)

        model.fit(xs, 
                  y_time, 
                  y_event, 
                  covariates_encoded=covariates_encoded,
                #   epochs=500, 
                #   instance_batch_size=3000, 
                #   lambdas=(0.0, 1.0),
                #   entropy_threshold=0.5, 
                  **kwargs)
        
        _, a, cell_hazards, cell_hazards_weighted = model.predict_cells(adata.obsm[pca_key])

        batch_lists = None
    elif feature_flavor == 'AE':
        # assert batch_key is None, 'Only support PCA for batch correction'
        # sc.pp.scale(adata, max_value=10)
        try:
            exp = adata.X.toarray()
        except:
            exp = adata.X

        if do_scale_ae:
            scaler = StandardScaler()
            # scaler = MinMaxScaler()
            exp = scaler.fit_transform(exp)
        else:
            scaler = None
        
        
        gamma_beta_weight = (0.1, 0)
        if rec_likelihood == 'ZIG':
            zero_prop = np.mean(exp == 0)
            if zero_prop > 0.9:
                print('Warning: the zero proportion exceeds 90%.')
                # gamma_beta_weight = (0.1, 0.1)

        if 'gamma_beta_weight' not in kwargs:
            kwargs['gamma_beta_weight'] = gamma_beta_weight

        xs = [exp[adata.obs[sample_column] == each] for each in patients]
        # xs = [adata[adata.obs[sample_column] == each].X for each in patients]
        input_size = xs[0].shape[1]

        if batch_key is not None:
            le = LabelEncoder()
            batchs = le.fit_transform(adata.obs[batch_key])
            batch_lists = [batchs[adata.obs[sample_column] == each] for each in patients]
            model = scSurvival(input_size, 
                               hidden_size=hidden_size, 
                               num_heads=num_heads, 
                               extract_feature=True, 
                               use_batch=True, num_batches=len(le.classes_), beta=beta, tau=tau, 
                               rec_distribution=rec_likelihood,
                               dropout=dropout,
                               covariate_size=covariate_size)
            model.le = le
        else:
            batchs = None
            batch_lists = None
            model = scSurvival(input_size, 
                               hidden_size=hidden_size, 
                               num_heads=num_heads, 
                               extract_feature=True, 
                               use_batch=False, 
                               beta=beta, tau=tau, 
                               rec_distribution=rec_likelihood,
                               dropout=dropout,
                               covariate_size=covariate_size)
            
        
        if model_load_dir is not None:
            model.load_state_dict(torch.load(os.path.join(model_load_dir, 'model.pt'), map_location=model.device, weight_only=True))
            print('Model loaded from %s' % model_load_dir)

        try:    
            model.scaler = scaler
        except:
            model.scaler = None

        if 'entropy_threshold' not in kwargs:
            kwargs['entropy_threshold'] = 0.7


        gene_weights = calc_gene_weights(exp, alpha=gene_weight_alpha)
        gene_weights = torch.tensor(gene_weights).float().to(model.device)

        if rec_likelihood == 'ZIG':
            non_zero_vars = []
            for i in range(exp.shape[1]):
                tmp_x = exp[:, i]
                tmp_check = tmp_x > 0
                if tmp_check.max():
                    non_zero_var = np.var(tmp_x[tmp_x > 0])
                else:
                    non_zero_var = 0.0
                non_zero_vars.append(non_zero_var)
            model.cell_model.feature_extractor.recon_logvar.copy_(torch.tensor(non_zero_vars).float().to(model.device)) 
            model.cell_model.feature_extractor.recon_logvar.requires_grad = True

        model.fit(xs, 
                  y_time, 
                  y_event, 
                  covariates_encoded=covariates_encoded,
                  batch_lists=batch_lists,
                  feature_weights=gene_weights,
                #   epochs=500, 
                #   instance_batch_size=3000, 
                #   lambdas=(1.0, 1.0),
                #   entropy_threshold=0.7,
                  **kwargs)

        h, a, cell_hazards, cell_hazards_weighted = model.predict_cells(exp, batch_labels=batchs)
        adata.obsm['X_ae'] = h.cpu().detach().numpy()

    adata.obs['hazard'] = cell_hazards.cpu().detach().numpy()
    adata.obs['attention'] = a.cpu().detach().numpy()
    adata.obs['hazard_adj'] = cell_hazards_weighted.cpu().detach().numpy()

    print('Added hazard and attention to adata.obs.')

    patient_hazards = model.predict_samples(xs, 
                                            batch_lists=batch_lists, 
                                            covariates_encoded=covariates_encoded,
                                            n_MC=predict_nMC
                                            ).view(-1).cpu().detach().numpy()
    surv['patient_hazards'] = patient_hazards[0:len(patients_with_surv)]

    if len(patients_wo_surv) > 0:
        surv_wo = pd.DataFrame(index=patients_wo_surv, columns=['patient_hazards'])
        surv_wo['patient_hazards'] = patient_hazards[len(patients_with_surv):]
        surv_wo = surv_wo.reindex(columns=surv.columns)
        surv = pd.concat([surv, surv_wo], axis=0)

    print('Added patient_hazards to surv.')

    model.feature_flavor = feature_flavor
    model.cov_processor = cov_psr
    if model.use_batch:
        model.batch_key = batch_key

    if model_save_dir is not None:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        torch.save(model.state_dict(), os.path.join(model_save_dir, 'model.pt'))
        print('Model saved to %s' % model_save_dir)

    return adata, surv, model

def PredictIndSample(adata_new, adata=None, model=None, adata_genes=None, covariates_new=None, n_MC=64):
    '''
    Predict individual sample hazards using the trained model.
    Parameters
    ----------
    n_MC: number of Monte Carlo samples for prediction. only when `sample_balance` is True in training.
    '''
    if adata is None:
        assert model.feature_flavor == 'AE', 'adata is required for PCA feature flavor.'
        assert adata_genes is not None, 'adata_genes is required for AE feature flavor if adata is None.'

    if adata_genes is None:
        adata_genes = adata.var_names

    # adata_new: AnnData object with normalized expression data
    # assert model.use_batch == False, 'predicting individual sample is not supported for batch correction. Try to train a model jointly with the new sample.'

    if model.use_batch:
        assert model.feature_flavor == 'AE', 'Only support AE feature flavor for batch correction automatically.'
        
        assert model.batch_key in adata_new.obs.columns, 'batch_key %s not in adata_new.obs' % model.batch_key
        try:
            batches = model.le.transform(adata_new.obs[model.batch_key].values)
        except:
            print('Predicting individual sample is not supported for batch correction. Try to train a model jointly with the new sample.')
            raise ValueError('The batch labels in adata_new.obs[%s] do not match those in training data.' % model.batch_key)
    else:
        batches = None

    feature_flavor = model.feature_flavor
    missing_genes = list(set(adata_genes) - set(adata_new.var_names))
    print('gene missing rate: %.2f%%' % (len(missing_genes) * 100 / len(adata_genes)))

    try:
        exp_df = pd.DataFrame(adata_new.X.toarray(), columns=adata_new.var_names)
    except:
        exp_df = pd.DataFrame(adata_new.X, columns=adata_new.var_names)

    if len(missing_genes) > 0:
        exp_df = pd.concat([exp_df, pd.DataFrame(0, index=exp_df.index, columns=missing_genes)], axis=1)
    exp = exp_df.loc[:, adata_genes].values

    if 'X_pca_trans' not in adata_new.obsm.keys() and feature_flavor == 'PCA':
        mean = adata.var['mean'].values.reshape(1, -1)
        std = adata.var['std'].values.reshape(1, -1)

        exp = (exp - mean) / std
        exp = np.clip(exp, -np.inf, 10)

    if covariates_new is not None:
        assert model.cov_processor is not None, 'cov__processor is required for covariates_new.'
        covariates_new_encoded = model.cov_processor.transform(covariates_new)
    else:
        covariates_new_encoded = None
    if feature_flavor == 'Custom':
        exp_custom = adata_new.obsm[model.feature_key]
        _, a, cell_hazards, cell_hazards_weighted = model.predict_cells(exp_custom, batch_labels=batches)
        patient_hazards = model.predict_samples(
            [exp_custom], 
            batch_lists=None,
            covariates_encoded=covariates_new_encoded
            ).view(-1).cpu().detach().numpy()

    elif feature_flavor == 'PCA':
        if 'X_pca_trans' in adata_new.obsm.keys():
            exp_pca = adata_new.obsm['X_pca_trans']
        else:
            exp_pca = exp @ adata.varm['PCs']

        _, a, cell_hazards, cell_hazards_weighted = model.predict_cells(exp_pca, batch_labels=batches)
        patient_hazards = model.predict_samples(
            [exp_pca], 
            batch_lists=None,
            covariates_encoded=covariates_new_encoded
            ).view(-1).cpu().detach().numpy()
        
    elif feature_flavor == 'AE':
        # exp_scaled = exp
        if model.scaler is not None:
            exp = model.scaler.transform(exp)
        h, a, cell_hazards, cell_hazards_weighted = model.predict_cells(exp, batch_labels=batches)
        patient_hazards = model.predict_samples(
            [exp], 
            batch_lists=[batches] if batches is not None else None,
            covariates_encoded=covariates_new_encoded,
            n_MC=n_MC
            ).view(-1).cpu().detach().numpy()
        
        adata_new.obsm['X_ae'] = h.cpu().detach().numpy()

    adata_new.obs['hazard'] = cell_hazards.cpu().detach().numpy()
    adata_new.obs['attention'] = a.cpu().detach().numpy()
    adata_new.obs['hazard_adj'] = cell_hazards_weighted.cpu().detach().numpy()
    
    print('Added hazard and attention to adata_new.obs.')
    return adata_new, patient_hazards[0]

if __name__ == "__main__":
    # multiple instance learning dataset
    xs = [torch.randn(10, 128) for _ in range(32)]
    y_time = torch.randn(32, 1)
    y_event = torch.randint(0, 2, (32, 1))

    input_size = xs[0].shape[1]

    model = scSurvival(input_size, hidden_size=128)
    model.fit(xs, y_time, y_event, epochs=100, instance_batch_size=16)
