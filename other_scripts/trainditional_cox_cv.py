import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, GridSearchCV, StratifiedKFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from scipy.stats import percentileofscore
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
from copy import deepcopy
import warnings
# Suppress warnings from sksurv about convergence, etc.
warnings.filterwarnings("ignore", category=UserWarning)


def cindex_scorer(estimator, X, y):
    try:
        # 确保数据类型正确
        event = y["event"].astype(bool)
        time = y["time"].astype(float)
            
        # 获取预测值
        pred_risk = estimator.predict(X)
            
        # 检查预测值是否有效
        if np.any(np.isnan(pred_risk)) or np.any(np.isinf(pred_risk)):
            print(f"Warning: Invalid predictions detected")
            return 0.0
            
        # 检查是否有足够的事件进行计算
        if np.sum(event) < 2:
            print(f"Warning: Too few events ({np.sum(event)}) for C-index calculation")
            return 0.0
            
        # 计算 C-index
        cindex = concordance_index_censored(event, time, pred_risk.flatten())[0]

        # 检查结果是否有效
        if np.isnan(cindex):
            print(f"Warning: C-index is NaN")
            return 0.0

        print(f"C-index computed: {cindex:.4f}")        
        return float(cindex)
            
    except Exception as e:
        print(f"Error in cindex_scorer: {e}")
        return 0.0

def select_features_by_variance(x, n_top=2000):
    std = np.std(x, axis=0)
    top_indices = np.argsort(std)[-n_top:]  # Get indices of the top n features

    return top_indices

# X: ndarray (n_samples, n_features)
# y: structured array with fields ('event', 'time'), e.g. from sksurv.util import Surv; Surv.from_arrays(...)
# outer_mode: 'loo' for Leave-One-Out CV or 'kfold' for K-Fold CV
def nested_cv_coxnet(
    X, y,
    outer_mode='loo', # 'loo' | 'kfold' | 'kstfold'
    outer_splits=5,           # Used when outer_mode='kfold'
    inner_splits=5,
    alpha_grid=None,
    n_alphas=30,
    alpha_min_ratio=0.01,
    l1_ratio=0.5,
    random_state=42,
    n_jobs=-1,
    return_coef=True,
    select_features=False,
    n_top=2000,
    do_pca=False
):  
    if do_pca:
        assert X.shape[1] > 50, "PCA is only meaningful when n_features > 50."
        if X.shape[0] < 50:
            npcas = 20
        else:
            npcas = 50

    # 自定义 c-index scorer
    scoring = make_scorer(cindex_scorer, greater_is_better=True)

    input_alpha_grid = deepcopy(alpha_grid)

    # Define outer cross-validation strategy
    if outer_mode == 'loo':
        outer_cv = LeaveOneOut()
    elif outer_mode == 'kfold':
        outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    elif outer_mode == 'kstfold':
        outer_cv = StratifiedKFold(
            n_splits=outer_splits,
            shuffle=True,
            random_state=random_state
        )
    else:
        raise ValueError("outer_mode must be 'loo' or 'kfold'.")

    # Store results
    outer_fold_infos = []
    kfold_cindices = []           # Only used in kfold mode

    # In LOO mode, store predicted percentiles for all samples to compute a unified C-index
    if outer_mode == 'loo':
        n = len(y)
        unified_pred_percentiles = np.zeros(n, dtype=float)

    if return_coef:
        unified_coefs = []


    # Outer CV loop
    if outer_mode == 'kstfold':
        outer_cv_list = list(outer_cv.split(X, y["event"].astype(int)))
    else:
        outer_cv_list = list(outer_cv.split(X))

    for fold_idx, (tr, te) in enumerate(outer_cv_list, start=1):
        print(f'Outer fold {fold_idx}:')
        X_train, X_test = X[tr].copy(), X[te].copy()
        y_train, y_test = y[tr], y[te]

        if select_features:
            features = select_features_by_variance(X_train, n_top=n_top)
            X_train = X_train[:, features]
            X_test = X_test[:, features]

        if input_alpha_grid is None:
            assert n_alphas is not None, "Either alpha_grid or n_alphas must be provided."
            if do_pca:
                pilot = make_pipeline(
                    StandardScaler(),  # Standardize features
                    PCA(n_components=npcas, random_state=42),
                    CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, n_alphas=n_alphas, alpha_min_ratio=alpha_min_ratio)
                )
            else:
                pilot = make_pipeline(
                    StandardScaler(),  # Standardize features
                    CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, n_alphas=n_alphas, alpha_min_ratio=alpha_min_ratio)
                )
            pilot.fit(X_train, y_train)
            alpha_grid = pilot.named_steps.get('coxnetsurvivalanalysis').alphas_
            print(f'Fold {fold_idx}: alpha grid min={alpha_grid.min():.4f}, max={alpha_grid.max():.4f}')

        # ===== Inner CV: select alpha =====
        # inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
        inner_cv = StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=42
        )
        
        
        # CoxnetSurvivalAnalysis expects 'alphas' as a list of lists (each element is a path or a single value)
        # param_grid = {"coxnetsurvivalanalysis__alphas": [[a] for a in alpha_grid]}

        # grid = GridSearchCV(
        #     make_pipeline(
        #         StandardScaler(),  # Standardize features
        #         CoxnetSurvivalAnalysis(l1_ratio=l1_ratio)
        #     ),
        #     param_grid=param_grid,
        #     cv=inner_cv,
        #     scoring=scoring,
        #     n_jobs=n_jobs
        # )
        # grid.fit(X_train, y_train)
        # best_alpha = grid.best_params_["coxnetsurvivalanalysis__alphas"][0]
        print(f'  Starting inner CV for alpha selection...')
        if len(alpha_grid) == 1:
            best_alpha = alpha_grid[0]
            best_alpha_idx = 0 
        else:
            in_cindex_folds = []
            for in_tr, in_te in inner_cv.split(X_train, y_train["event"].astype(int)):
                X_in_tr, X_in_te = X_train[in_tr], X_train[in_te]
                y_in_tr, y_in_te = y_train[in_tr], y_train[in_te]

                try:
                    if do_pca:
                        in_model = make_pipeline(
                            StandardScaler(),
                            PCA(n_components=npcas, random_state=42),
                            CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=alpha_grid)
                        )
                    else:
                        in_model = make_pipeline(
                            StandardScaler(),
                            CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=alpha_grid)
                        )

                    in_model.fit(X_in_tr, y_in_tr)
                    in_cindexs = []

                    for _, alpha in enumerate(alpha_grid):
                        risk_scores = in_model.predict(X_in_te, alpha=alpha)
                        # Compute C-index
                        try:
                            cindex = concordance_index_censored(
                                y_in_te["event"].astype(bool),
                                y_in_te["time"].astype(float),
                                risk_scores
                            )[0]
                        except Exception as e:
                            cindex = np.nan

                        in_cindexs.append(cindex)
                except Exception as e:
                    # raise e
                    in_cindexs = []
                    for alpha in alpha_grid:
                        try:
                            if do_pca:
                                in_model = make_pipeline(
                                    StandardScaler(),
                                    PCA(n_components=npcas, random_state=random_state),
                                    CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=[alpha])
                                )
                            else:
                                in_model = make_pipeline(
                                    StandardScaler(),
                                    CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=[alpha])
                                )
                            in_model.fit(X_in_tr, y_in_tr)
                            risk_scores = in_model.predict(X_in_te)
                            try:
                                cindex = concordance_index_censored(
                                    y_in_te["event"].astype(bool),
                                    y_in_te["time"].astype(float),
                                    risk_scores
                                )[0]
                            except Exception as e:
                                cindex = np.nan
                            in_cindexs.append(cindex)
                        except Exception as e:
                            in_cindexs.append(0.0)
                        
                in_cindex_folds.append(in_cindexs)
            
            # Average C-index across inner folds for each alpha
            in_cindex_folds = np.array(in_cindex_folds, dtype=float)  # shape: (n_inner_folds, n_alphas)
            mean_in_cindexs = np.nanmean(in_cindex_folds, axis=0)
            # print(f"Fold {fold_idx}: Inner mean C-index per alpha: {mean_in_cindexs}")
            # print(f"Fold {fold_idx}: Alpha grid: {alpha_grid}")

            best_alpha_idx = int(np.nanargmax(mean_in_cindexs))
            best_alpha = alpha_grid[best_alpha_idx]

        # print(f"Fold {fold_idx}: Selected best alpha = {best_alpha} with mean C-index = {mean_in_cindexs[best_alpha_idx]:.4f}")

        # ===== Retrain on the entire training set with the best alpha =====
        print(f'  Retraining on the entire training set with alpha={best_alpha}...')
        alpha_grid_sorted = np.sort(alpha_grid)
        for alpha_idx in range(best_alpha_idx, len(alpha_grid)):
            try:
                best_alpha = alpha_grid_sorted[alpha_idx]
                if do_pca:
                    model = make_pipeline(
                        StandardScaler(),  # Standardize features
                        PCA(n_components=npcas, random_state=42),
                        CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=[best_alpha])
                    )
                else:
                    model = make_pipeline(
                        StandardScaler(),  # Standardize features
                        CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=[best_alpha])
                    )
                model.fit(X_train, y_train)
                # print(f"Successfully trained with alpha={best_alpha}")
                break  # Successfully trained
            except Exception as e:
                print(f"Warning: Failed to fit model with alpha={best_alpha}, trying next alpha.")

        train_scores = model.predict(X_train, alpha=best_alpha)
        train_cindex = concordance_index_censored(
            y_train["event"].astype(bool),
            y_train["time"].astype(float),
            train_scores
        )[0]

        if return_coef:
            try:
                coxnet_model = model.named_steps.get('coxnetsurvivalanalysis') or model[-1]
            except KeyError:
                coxnet_model = model[-1]
                
            coef = coxnet_model.coef_.copy()
            if coef.ndim > 1:  # If it's a 1D array, reshape to 2D
                coef = coef.reshape(-1)
            unified_coefs.append(coef)

        # ===== Outer evaluation =====
        if outer_mode == 'loo':
            # 1) Get raw scores on the training set
            train_scores = model.predict(X_train)
            # 2) Raw score for the single test sample
            test_score = float(model.predict(X_test))
            # 3) Convert test score to percentile relative to the training set
            pct = percentileofscore(train_scores, test_score, kind='rank')
            unified_pred_percentiles[te[0]] = pct

            outer_fold_infos.append({
                "fold": fold_idx,
                "best_alpha": best_alpha,
                "train_cindex": train_cindex,
                "test_index": int(te[0]),
                "test_score": test_score,
                "percentile_in_train": pct
            })
        else:  # 'kfold'
            test_scores = model.predict(X_test)
            cindex = concordance_index_censored(
                y_test["event"], y_test["time"], test_scores
            )[0]
            kfold_cindices.append(cindex)
            outer_fold_infos.append({
                "fold": fold_idx,
                "best_alpha": best_alpha,
                "train_cindex": train_cindex,
                "cindex": cindex,
                "n_test": len(te)
            })

    # ===== Summary =====
    if outer_mode == 'loo':
        # Compute unified C-index using percentiles as predicted scores
        events_all = y["event"].astype(bool)
        times_all = y["time"].astype(float)
        unified_cindex = concordance_index_censored(
            events_all, times_all, unified_pred_percentiles
        )[0]

        summary = {
            "outer_mode": "loo",
            "unified_cindex": unified_cindex,
            "details": outer_fold_infos
        }
    else:
        mean_ci = float(np.mean(kfold_cindices))
        std_ci = float(np.std(kfold_cindices, ddof=1)) if len(kfold_cindices) > 1 else 0.0
        summary = {
            "outer_mode": "kfold",
            "mean_cindex": mean_ci,
            "std_cindex": std_ci,
            "details": outer_fold_infos
        }
    
    if return_coef:
        unified_coefs = np.array(unified_coefs, dtype=float)
        summary["unified_coefs"] = unified_coefs.mean(axis=0)  # Average coefficients across folds
        if unified_coefs.shape[0] > 1:
            summary["coef_std"] = unified_coefs.std(axis=0)
    
    print('Nested CV completed.')
    return summary


# --- Helper: convert RSF survival functions to a monotone risk score ---
# We use negative expected survival time: risk = -∫ S(t) dt
# This yields higher risk for curves that drop earlier (worse survival).
def _rsf_risk_scores(model: RandomSurvivalForest, X):
    surv_funcs = model.predict_survival_function(X)
    # Integrate each survival curve on its own time grid
    risks = []
    for sf in surv_funcs:
        # sf.x: times, sf.y: survival probabilities
        t = np.asarray(sf.x, dtype=float)
        s = np.asarray(sf.y, dtype=float)
        if t.size < 2:
            risks.append(0.0)
            continue
        # trapezoidal integration
        area = np.trapz(s, t)
        risks.append(-area)
    return np.array(risks, dtype=float)

def cv_rsf(
    X, y,
    outer_mode='loo',          # 'loo' or 'kfold' or 'kstfold'
    outer_splits=5,            # used when outer_mode='kfold'
    random_state=42,
    n_jobs=-1,
    # --- RSF hyperparameters (fixed; no tuning in inner loop) --- set to the default in RSF except max_features 
    n_estimators=100,
    min_samples_split=6,
    min_samples_leaf=3,
    max_features="sqrt",
    bootstrap=True,
    select_features=False,
    n_top=2000,
    do_pca=False
):
    """
    Nested-CV style evaluation using Random Survival Forest without inner tuning.
    - LOO: transform each test score to percentile w.r.t. the training scores of that fold,
           then compute a unified C-index once at the end.
    - K-fold: compute a C-index per fold on raw RSF risk scores, then report mean/std.
    """

    if do_pca:
        assert X.shape[1] > 50, "PCA is only meaningful when n_features > 50."
        if X.shape[0] < 50:
            npcas = 20
        else:
            npcas = 50
        
    # ----- choose outer splitter -----
    if outer_mode == 'loo':
        outer_cv = LeaveOneOut()
    elif outer_mode == 'kfold':
        outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    elif outer_mode == 'kstfold':
        outer_cv = StratifiedKFold(
            n_splits=outer_splits,
            shuffle=True,
            random_state=random_state
        )
    else:
        raise ValueError("outer_mode must be 'loo' or 'kfold'")

    # ----- containers -----
    fold_infos = []
    if outer_mode == 'loo':
        n = len(y)
        unified_percentiles = np.zeros(n, dtype=float)
    else:
        fold_cindices = []

    # ----- outer loop -----
    if outer_mode == 'kstfold':
        outer_cv_list = list(outer_cv.split(X, y["event"].astype(int)))
    else:
        outer_cv_list = list(outer_cv.split(X))
        
    for fold, (tr, te) in enumerate(outer_cv_list, start=1):
        X_train, X_test = X[tr].copy(), X[te].copy()
        y_train, y_test = y[tr], y[te]

        if select_features:
            features = select_features_by_variance(X_train, n_top=n_top)
            X_train = X_train[:, features]
            X_test = X_test[:, features]

        if do_pca:
            pca = PCA(n_components=npcas, random_state=42)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        # ----- fit RSF on the whole outer training set (no inner tuning) -----
        rsf = RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        rsf.fit(X_train, y_train)

        if outer_mode == 'loo':
            # training distribution for percentile transform
            train_scores = _rsf_risk_scores(rsf, X_train)
            test_score = float(_rsf_risk_scores(rsf, X_test))
            pct = percentileofscore(train_scores, test_score, kind='rank')  # 0..100
            unified_percentiles[te[0]] = pct

            fold_infos.append({
                "fold": fold,
                "test_index": int(te[0]),
                "percentile_in_train": pct,
            })
        else:
            test_scores = _rsf_risk_scores(rsf, X_test)
            cindex = concordance_index_censored(
                y_test["event"].astype(bool),
                y_test["time"].astype(float),
                test_scores
            )[0]
            fold_cindices.append(float(cindex))
            fold_infos.append({
                "fold": fold,
                "cindex": float(cindex),
                "n_test": int(len(te)),
            })

    # ----- summary -----
    if outer_mode == 'loo':
        events_all = y["event"].astype(bool)
        times_all = y["time"].astype(float)
        unified_cindex = concordance_index_censored(
            events_all, times_all, unified_percentiles
        )[0]
        return {
            "outer_mode": "loo",
            "unified_cindex": float(unified_cindex),
            "details": fold_infos
        }
    else:
        mean_ci = float(np.mean(fold_cindices))
        std_ci = float(np.std(fold_cindices, ddof=1)) if len(fold_cindices) > 1 else 0.0
        return {
            "outer_mode": "kfold",
            "mean_cindex": mean_ci,
            "std_cindex": std_ci,
            "details": fold_infos
        }

# -----------------------
# Example usage
# -----------------------
# from sksurv.util import Surv
# y = Surv.from_arrays(event=event_bool_array, time=time_float_array)
# res_loo = nested_cv_rsf(X, y, outer_mode='loo')
# res_kf  = nested_cv_rsf(X, y, outer_mode='kfold', outer_splits=5)

# =====================
# Example usage (assuming X, y are prepared)
# =====================
# from sksurv.util import Surv
# y = Surv.from_arrays(event=event_bool_array, time=time_float_array)

# 1) Outer LOO CV + unified C-index
# res_loo = nested_cv_coxnet(X, y, outer_mode='loo', inner_splits=5)

# 2) Outer 5-fold CV + mean/std C-index
# res_kf  = nested_cv_coxnet(X, y, outer_mode='kfold', outer_splits=5, inner_splits=5)
