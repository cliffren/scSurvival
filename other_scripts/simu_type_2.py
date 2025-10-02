#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
os.environ["RPY2_CFFI_MODE"] = "ABI"
import rpy2.robjects as ro
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
rpy2_logger.setLevel("ERROR")   # 只显示错误，屏蔽 message 和 warning
rpy2_logger.propagate = False   # 阻止继续传给 root logger

import sys
sys.path.append("../../")

# from scSurvival.scsurvival import scSurvival, scSurvivalRun, PredictIndSample
from scSurvival_beta import scSurvivalRun, PredictIndSample
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from tqdm import tqdm, trange
import scanpy as sc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import io
import contextlib
f = io.StringIO()
from lifelines.utils import concordance_index
from scipy.stats import percentileofscore
from utils import *


# In[7]:


def load_r_packages():
    ro.r('''
    rm(list=ls())
    library("splatter")
    library(scran)
    library(Seurat)
    # library(preprocessCore)
    library(pROC)
    # library(APML2)
    # library(APML1)
    # library(APML0)

    library(ggplot2)
    library(dplyr)
    library(caret)
    set.seed(1)
    ''')


def simulated_base_sc_dataset(seed=42, plot=False):
    ro.r(f'''
    seed <- {seed}
    sim.groups <- splatSimulateGroups(batchCells = c(6000, 4000), nGenes=5000,
                                  #group.prob = c(0.9, 0.05, 0.05),
                                  group.prob = c(0.7, 0.15, 0.15),
                                  de.prob = c(0.2, 0.06, 0.06), de.facLoc = c(0.1, 0.1, 0.1),
                                  de.facScale = 0.4,
                                  seed = seed)#

    data <- CreateSeuratObject(counts = counts(sim.groups), project = 'Scissor_Single_Cell')
    data <- AddMetaData(object = data, metadata = sim.groups$Group, col.name = "sim.group")
    data$Actual.cond <- recode(data$sim.group,'Group1'='other', 'Group2'='good.survival', 'Group3'='bad.survival')
    data$batch <- c(rep('Batch1', 6000), rep('Batch2', 4000))


    select_gene_ids <- 1:2000
    data <- NormalizeData(object = data, normalization.method = "LogNormalize", 
                          scale.factor = 10000)
    data <- FindVariableFeatures(object = data, selection.method = 'vst', nfeatures=2000)
    var_features_genes = VariableFeatures(data)
    
    # data -> data.raw
    ''')

    if plot:
        ro.r('''
        data <- ScaleData(object = data)
        data <- RunPCA(object = data, features = VariableFeatures(data)[select_gene_ids])

        data <- RunUMAP(object = data, dims = 1:10, n.neighbors = 5, min.dist=0.5, spread=1.5)
        # data <- RunUMAP(object = data, dims = 1:10)
        data <- FindNeighbors(object = data, dims = 1:10, k.param=20)
        # data <- FindNeighbors(object = data, dims = 1:10, k.param=20,prune.SNN = 0.2)
        data <- FindClusters(object = data, resolution = 0.5)

        DimPlot(object = data, reduction = 'umap', group.by = 'seurat_clusters', label = F, label.size = 10,pt.size=0.5)
        ggsave(paste0(save_path, 'simu_seurat_cluster_umap.pdf'), height = 5, width = 7)

        DimPlot(object = data, reduction = 'umap', group.by = 'sim.group', pt.size = 0.5, label = T)
        ggsave(paste0(save_path, 'simu_group_umap.pdf'), height = 5, width = 7)

        # DimPlot(object = data, reduction = 'umap',  cols = c('grey','blue', 'red'), group.by = 'sim.group', pt.size = 0.5, label = T)
        # 
        DimPlot(object = data, reduction = 'umap',  cols = c('grey','blue', 'red'), group.by = 'Actual.cond', pt.size = 0.5, label = T)
        ggsave(paste0(save_path, 'simu_surv_group_umap.pdf'), height = 5, width = 7)

        DimPlot(object = data, reduction = 'umap', group.by = 'batch', cols = c('red','blue'), pt.size = 0.5, label = F)
        ggsave(paste0(save_path, 'simu_batch_umap.pdf'), height = 5, width = 7)
        ''')

def simu_celldiff_across_batches(ds_rate=0.7, plot=False):
    ro.r(f'''
    # data.raw -> data
    ds_rate <- {ds_rate}
    set.seed(seed)

    batch1_good_cells <- colnames(data)[data$batch=='Batch1' & data$Actual.cond=='good.survival']
    batch2_bad_cells <- colnames(data)[data$batch=='Batch2' & data$Actual.cond=='bad.survival']

    num_b1g_sample = floor(length(batch1_good_cells) * (1 - ds_rate))
    num_b2b_sample = floor(length(batch2_bad_cells) * (1 - ds_rate))

    dn_sampled_cells <- c(sample(batch1_good_cells, num_b1g_sample), sample(batch2_bad_cells, num_b2b_sample))


    cells_kept <- setdiff(colnames(data), dn_sampled_cells)
    data <- data[, cells_kept]
    ''')

    if plot:
        ro.r('''
        DimPlot(object = data, reduction = 'umap', group.by = 'sim.group', pt.size = 0.5, label = T)
        ggsave(paste0(save_path, 'simu_group_umap_dnsample.pdf'), height = 5, width = 7)

        DimPlot(object = data, reduction = 'umap',  cols = c('grey','blue', 'red'), group.by = 'Actual.cond', pt.size = 0.5, label = T)
        ggsave(paste0(save_path, 'simu_surv_group_umap_dnsample.pdf'), height = 5, width = 7)

        DimPlot(object = data, reduction = 'umap', group.by = 'batch', cols = c('red','blue'), pt.size = 0.5, label = F)
        ggsave(paste0(save_path, 'simu_batch_umap_dnsample.pdf'), height = 5, width = 7)
        ''')
    
    
def simulated_sc_datasets():
    ro.r('''
    Expression_pbmc <- as.matrix(data@assays[["RNA"]]@layers[["data"]])
    rownames(Expression_pbmc) <- rownames(data)
    colnames(Expression_pbmc) <- colnames(data)
    Expression_pbmc <- as.data.frame(Expression_pbmc)
    all_genes <- rownames(Expression_pbmc)
         
    set.seed(seed)
    sampled_cells = 1000
    bulk_num=50
    
    sc_data_list = list()

    ###---simulation from batch1---------------------
    print('Simulating from Batch1 ...')
    other_cells <- colnames(Expression_pbmc)[data$Actual.cond=='other' & data$batch=='Batch1']
    good_cells <- colnames(Expression_pbmc)[data$Actual.cond=='good.survival' & data$batch=='Batch1']
    bad_cells <- colnames(Expression_pbmc)[data$Actual.cond=='bad.survival' & data$batch=='Batch1']
    num_good <- length(good_cells)
    num_bad <- length(bad_cells)

    bulk_condition = NULL
    censor_prob = 0.1

    status = NULL
    surv_time = NULL

    num_good_cond_cells = NULL
    num_bad_cond_cells = NULL

    pb <- txtProgressBar(min = 1, max = bulk_num, style = 3)
    for (i in 1:bulk_num){
      setTxtProgressBar(pb, i)
      ratio <- (i-1) / (bulk_num-1)
      # ratio <- plogis((ratio - 0.5) * 2 * 6)
      num_good_cond_cells_i = round(num_good * ratio)
      num_bad_cond_cells_i = round(num_bad * (1-ratio))
      condition_good_cells <- good_cells[sample(num_good, num_good_cond_cells_i , replace=TRUE)]
      condition_bad_cells <- bad_cells[sample(num_bad, num_bad_cond_cells_i, replace=TRUE)]
      condition_cells <- c(condition_good_cells, condition_bad_cells, other_cells)
      # condition_cells <- c(condition_bad_cells, other_cells)
  
      num_good_cond_cells = c(num_good_cond_cells, num_good_cond_cells_i)
      num_bad_cond_cells = c(num_bad_cond_cells, num_bad_cond_cells_i)
  
      Expression_condition = Expression_pbmc[, condition_cells]
      Expression_selected <- Expression_condition[, sample(ncol(Expression_condition),size=sampled_cells,replace=TRUE)]
  
      sc_data_list[[sprintf('bulk%d', i)]] <- Expression_selected
  
      if (runif(1, min = 0, max = 1) < censor_prob){
        status = c(status, 0)
        surv_time = c(surv_time, sample(i, 1))
      }
      else{
        surv_time = c(surv_time, i)
        status = c(status, 1)
      }
    }


    ###---simulation from batch2---------------------
    print('\n')
    print('Simulating from Batch2 ...')
    other_cells <- colnames(Expression_pbmc)[data$Actual.cond=='other' & data$batch=='Batch2']
    good_cells <- colnames(Expression_pbmc)[data$Actual.cond=='good.survival' & data$batch=='Batch2']
    bad_cells <- colnames(Expression_pbmc)[data$Actual.cond=='bad.survival' & data$batch=='Batch2']
    num_good <- length(good_cells)
    num_bad <- length(bad_cells)

    censor_prob = 0.1
    pb <- txtProgressBar(min = 1, max = bulk_num, style = 3)
    for (i in 1:bulk_num){
      setTxtProgressBar(pb, i)
      ratio <- (i-1) / (bulk_num-1)
      # ratio <- plogis((ratio - 0.5) * 2 * 6)
      num_good_cond_cells_i = round(num_good * ratio)
      num_bad_cond_cells_i = round(num_bad * (1-ratio))
      condition_good_cells <- good_cells[sample(num_good, num_good_cond_cells_i , replace=TRUE)]
      condition_bad_cells <- bad_cells[sample(num_bad, num_bad_cond_cells_i, replace=TRUE)]
      condition_cells <- c(condition_good_cells, condition_bad_cells, other_cells)
      # condition_cells <- c(condition_bad_cells, other_cells)
  
      num_good_cond_cells = c(num_good_cond_cells, num_good_cond_cells_i)
      num_bad_cond_cells = c(num_bad_cond_cells, num_bad_cond_cells_i)
  
      Expression_condition = Expression_pbmc[, condition_cells]
      Expression_selected <- Expression_condition[, sample(ncol(Expression_condition),size=sampled_cells,replace=TRUE)]
  
      sc_data_list[[sprintf('bulk%d', i+bulk_num)]] <- Expression_selected
  
      if (runif(1, min = 0, max = 1) < censor_prob){
        status = c(status, 0)
        surv_time = c(surv_time, sample(i, 1))
      }
      else{
        surv_time = c(surv_time, i)
        status = c(status, 1)
      }
    }

    bulk_names <- paste0('bulk', 1:(bulk_num*2))
    surv_info <- data.frame(
      time=surv_time,
      status=status,
      num.good.cells = num_good_cond_cells,
      num.bad.cells = num_bad_cond_cells,
      row.names = bulk_names
    )

    dim(surv_info)
    dim(Expression_pbmc)
    
         
    labels <- data$Actual.cond
    labels <- as.data.frame(labels)
    row.names(labels) <- colnames(data)
         
    batches <- as.data.frame(data$batch)
    row.names(batches) <- colnames(data)
    colnames(batches) <- 'data.batch'
    ''')

    # collected sc_data_list, surv_info, Expression_pbmc and transfer to python
    surv_info_df = r_to_pandas("surv_info")
    Expression_pbmc_df = r_to_pandas("Expression_pbmc")
    sc_data_list = r_list_to_pydict_df("sc_data_list")  # dict: { 'bulk_1': DataFrame, ... }
    labels_df = r_to_pandas("labels")
    batches_df = r_to_pandas("batches")

    features = {
        'all_genes': list(ro.r("all_genes")),
        'hvg': list(ro.r("var_features_genes"))
    }

    return_data = {
        'sc_data_list': sc_data_list,
        'surv_info_df': surv_info_df,
        'Expression_pbmc_df': Expression_pbmc_df,
        'labels_df': labels_df,
        'batches_df': batches_df,
        'features': features
    }

    return return_data


# In[8]:


def organize_data_for_model(datasets):
    sc_data_list = datasets['sc_data_list']
    clinic = datasets['surv_info_df']

    xs = []
    samples = []
    batches = []
    for i, (key, val) in enumerate(tqdm(sc_data_list.items())):
        df = val
        xs.append(df.values.T)
        samples.extend([key] * df.shape[1])

        if i < 50:
            batches.extend(['Batch1'] * df.shape[1])
        else:
            batches.extend(['Batch2'] * df.shape[1])

    obs_df = pd.DataFrame({'sample': samples, 'batch': batches})
    obs_df.index = np.arange(len(samples))

    X = np.concatenate(xs, axis=0)
    adata = sc.AnnData(X, obs=obs_df, var=pd.DataFrame(index=datasets['features']['all_genes']))

    adata.raw = adata.copy()
    adata = adata[:, datasets['features']['hvg']]

    surv = clinic[['time', 'status']].copy()
    surv['time'] = surv['time'].astype(float)
    surv['status'] = surv['status'].astype(int)

    df = datasets['Expression_pbmc_df']
    x = df.values.T
    sim_group = datasets['labels_df']
    sim_group = sim_group['labels'].values

    adata_new = sc.AnnData(x, obs=pd.DataFrame(sim_group, index=np.arange(x.shape[0]), columns=['sim_group']), var=pd.DataFrame(index=datasets['features']['all_genes']))
    adata_new.obs['batch'] = datasets['batches_df']['data.batch'].values

    return adata, surv, adata_new

def detect_subpopulations(adata, surv, adata_new, entropy_threshold=0.7):
    adata, surv, model = scSurvivalRun(adata, 
        sample_column='sample',
        surv=surv,
        batch_key='batch',
        feature_flavor='AE',
        entropy_threshold=entropy_threshold,
        lambdas=(0.01, 1.0),
        pretrain_epochs=200,
        epochs=500,
        weight_decay=0.01,
        lr=0.001,
        patience=100,
        rec_likelihood='ZIG',
        do_scale_ae=False,
        beta=0.1, tau=0.2, 
        sample_size_ae=None,
        finetue_lr_factor=0.1,
        gene_weight_alpha=0.2,
        gamma_beta_weight=(0.1, 0.0),
        once_load_to_gpu=True,
        use_amp=False,
        fitnetune_strategy='alternating', # jointly, alternating, alternating_lightly,
        )

    data = adata.obs['attention'].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(data)
    cluster_centers = kmeans.cluster_centers_
    atten_thr = cluster_centers.flatten().mean()
    
    adata_new, _ = PredictIndSample(adata_new, adata, model)

    attention = adata_new.obs['attention'].values
    hazard_adj = adata_new.obs['hazard_adj'].values
    hazard = adata_new.obs['hazard'].values

    risk_group = np.array(['inattentive'] * attention.shape[0], dtype=object)
    risk_group[np.logical_and(attention >= atten_thr, hazard_adj > 0)] = 'higher'
    risk_group[np.logical_and(attention >= atten_thr, hazard_adj <= 0)] = 'lower'

    # higher -> bad.survival, lower -> good.survival, inattentive -> other 

    risk_group_recoded = np.array(['other'] * attention.shape[0], dtype=object)
    risk_group_recoded[risk_group == 'higher'] = 'bad.survival'
    risk_group_recoded[risk_group == 'lower'] = 'good.survival'

    clf_report = classification_report(adata_new.obs['sim_group'].values, risk_group_recoded, output_dict=True, zero_division=0)

    clf_report_df = pd.DataFrame(clf_report).T
    return clf_report_df, adata_new

def cross_validation_samples(adata, surv, entropy_threshold=0.7):
    # 交叉验证样本
    adata = adata.raw.to_adata()
    adata.obs['patient_no'] = adata.obs['sample']
    patients = adata.obs['patient_no'].unique()

    # K fold cross validation
    cv_hazards_adj_cells = np.zeros((adata.shape[0], ))
    surv['cv_hazards_adj_patient'] = 0.0
    surv['cv_hazard_percentile_patient'] = 0.0
    cindexs = []
    surv_test_all_folds = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(kf.split(patients)):

        print(f'fold {i}, train_size: {train_index.shape[0]}, test_size: {test_index.shape[0]}')
        train_patients = patients[train_index]
        test_patients = patients[test_index]

        # train
        adata_train = adata[adata.obs['patient_no'].isin(train_patients), :].copy()
    
        ## select HVGs on training set only
        sc.pp.highly_variable_genes(adata_train, n_top_genes=2000, subset=False, flavor='seurat')
        hvgs = adata_train.var[adata_train.var['highly_variable']].index.tolist() 
        adata_train = adata_train[:, hvgs]

        surv_train = surv.loc[surv.index.isin(train_patients), :].copy()

        adata_train, surv_train, model = scSurvivalRun(
            adata_train,
            sample_column='sample',
            surv=surv_train,
            batch_key='batch',
            feature_flavor='AE',
            entropy_threshold=entropy_threshold,
            validate=True,
            validate_ratio=0.2,
            validate_metric='ccindex',
            lambdas=(0.01, 1.0),
            pretrain_epochs=200,
            epochs=500,
            weight_decay=0.01,
            lr=0.001,
            patience=100,
            rec_likelihood='ZIG',
            do_scale_ae=False,
            beta=0.1, tau=0.2, 
            sample_size_ae=None,
            finetue_lr_factor=0.1,
            gene_weight_alpha=0.2,
            gamma_beta_weight=(0.1, 0.0),
            once_load_to_gpu=True,
            use_amp=False,
            fitnetune_strategy='alternating', # jointly, alternating, alternating_lightly,
            )  
        
        
        train_cindex = concordance_index(surv_train['time'], -surv_train['patient_hazards'], surv_train['status'])
        print(f'train c-index: {train_cindex:.4f}')

        # test
        print('testing...')
        adata_test = adata[adata.obs['patient_no'].isin(test_patients), :].copy()
        adata_test = adata_test[:, hvgs]

        with contextlib.redirect_stdout(f):
            for test_patient in test_patients:
                adata_test_patient = adata_test[adata_test.obs['patient_no'] == test_patient, :].copy()
                adata_test_patient, patient_hazard = PredictIndSample(adata_test_patient, adata_train, model)
                cv_hazards_adj_cells[adata.obs['patient_no'] == test_patient] = adata_test_patient.obs['hazard_adj'].values
                surv.loc[surv.index == test_patient, 'cv_hazards_adj_patient'] = patient_hazard
                surv.loc[surv.index == test_patient, 'cv_hazard_percentile_patient'] = percentileofscore(surv_train['patient_hazards'], patient_hazard, kind='rank')

        surv_test = surv.loc[surv.index.isin(test_patients), :]
        c_index = concordance_index(surv_test['time'], -surv_test['cv_hazards_adj_patient'], surv_test['status'])

        cindexs.append(c_index)
        surv_test_all_folds.append(surv_test)

        print(f'c-index: {c_index:.4f}')
        print('='*50)

        # if i == 0:
        #     break

    mean_cindex = np.mean(cindexs)
    std_cindex = np.std(cindexs)

    print(f'mean c-index: {mean_cindex:.4f} ± {std_cindex:.4f}')
    cindexs_df = pd.DataFrame(cindexs, columns=['c-index'], index=['fold%d' % i for i in range(5)])

    cindex_results = {
        'mean_cindex': mean_cindex,
        'std_cindex': std_cindex,
        'cindexs_df': cindexs_df
    }

    return cindex_results

def create_data(seed=42, ds_rate=0.3):
    load_r_packages()
    simulated_base_sc_dataset(seed=seed, plot=False)
    simu_celldiff_across_batches(ds_rate=ds_rate, plot=False)
    datasets = simulated_sc_datasets()
    adata, surv, adata_new = organize_data_for_model(datasets)
    return adata, surv, adata_new