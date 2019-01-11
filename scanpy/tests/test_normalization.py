import numpy as np
from scipy import sparse as sp
import scanpy.api as sc
from anndata import AnnData

A = np.array([
    [1., 0., 1.],
    [3., 0., 1.],
    [5., 6., 1.]
])

A_res = np.array([
    [1.,         0.,         1.        ],
    [3.,         0.,         1.        ],
    [0.71428573, 0.85714287, 0.14285715]
])

def test_normalize_total():
    adata = AnnData(sp.csr_matrix(A), layers={'L':A*2})
    sc.pp.normalize_total(adata, layers='all')
    med = np.median(A.sum(1))
    assert np.allclose(adata.X.toarray().sum(1), med)
    med_L = np.median((A*2).sum(1))
    assert np.allclose(adata.layers['L'].sum(1), med_L)
    sc.pp.normalize_total(adata, target_sum=3, layer_norm='after', layers='all')
    assert np.allclose(adata.X.toarray().sum(1), 3)
    assert np.allclose(adata.layers['L'].sum(1), 3)

def test_normalize_quantile():
    adata = AnnData(sp.csr_matrix(A))
    sc.pp.normalize_quantile(adata, quantile=0.7)
    assert np.allclose(adata.X.toarray(), A_res)
