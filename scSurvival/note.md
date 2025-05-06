adata: raw counts 
1. feature filter; -> features
2. normalize;
3. HVG 2000; -> hvgs
4. scale;
5. pca.

adata_test: raw counts
1. adata_test[:, np.intersect(features, adata_test.var_names)];
2. normalize;
3. padding missing hvgs;
4. adata_test[:, hvgs];
5. scale_trans;
6. pca_trans.

