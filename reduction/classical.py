from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap

def apply_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca.explained_variance_ratio_

def apply_tsne(X, n_components=2):
    tsne = TSNE(n_components=n_components)
    X_reduced = tsne.fit_transform(X)
    return X_reduced
