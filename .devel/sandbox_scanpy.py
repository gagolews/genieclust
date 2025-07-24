# Based on the code contributed by @gokceneraslan
# see https://github.com/gagolews/genieclust/issues/71


# pip install scanpy
# pip install louvain
# pip install scikit-misc


import scanpy as sc
sc.set_figure_params(dpi=100)
ad = sc.datasets.pbmc3k()

sc.pp.filter_genes(ad, min_cells=10)
ad.layers['counts'] = ad.X.copy()
sc.pp.normalize_total(ad, target_sum=10000)
sc.pp.log1p(ad)
sc.pp.highly_variable_genes(ad, n_top_genes=1000, flavor='seurat_v3', subset=True, layer='counts')
sc.pp.scale(ad, max_value=8)
sc.pp.pca(ad)

sc.pp.neighbors(ad)

sc.tl.umap(ad)
sc.tl.louvain(ad, resolution=0.2)

X_hidim = ad.X
X_lodim = ad.obsm['X_pca']


import genieclust
import numpy as np

g = genieclust.Genie(n_clusters=4, metric='cosine', M=25, postprocess="all")
labels = g.fit_predict(X_hidim)
ad.obs['genie_labels'] = labels.astype(str)
sc.pl.umap(ad, color='genie_labels')

g = genieclust.Genie(n_clusters=3, metric='cosine')
labels = g.fit_predict(X_lodim)
ad.obs['genie_labels'] = labels.astype(str)
sc.pl.umap(ad, color='genie_labels')

sc.pl.umap(ad, color='louvain')


mst = genieclust.internal.mst_from_distance(X_hidim)
genieclust.plots.plot_segments(mst[1], ad.obsm["X_umap"])


X_hidim_std = (X_hidim-X_hidim.mean(axis=0))/(X_hidim.std(axis=0, ddof=1))
g = genieclust.Genie(n_clusters=3)
labels = g.fit_predict(X_hidim_std)
ad.obs['genie_labels_std'] = labels.astype(str)
sc.pl.umap(ad, color='genie_labels_std')


mst = genieclust.internal.mst_from_distance(X_hidim_std)
genieclust.plots.plot_segments(mst[1], ad.obsm["X_umap"])



mst = genieclust.internal.mst_from_distance(np.array(X_lodim, copy=True, order='C'))
#sc.pl.umap(ad, color='genie_labels')
genieclust.plots.plot_segments(mst[1], ad.obsm["X_umap"])


from sklearn.decomposition import PCA
g = genieclust.Genie(n_clusters=3)
labels = g.fit_predict(PCA(20).fit_transform(X_hidim))
ad.obs['genie_labels'] = labels.astype(str)
sc.pl.umap(ad, color='genie_labels')

