import numpy as np

print('=' * 60)
print('TEST 1: tabpfn_extensions.TabPFNClassifier (no model_path)')
print('=' * 60)
try:
    from tabpfn_extensions import TabPFNClassifier as Clf1
    from tabpfn_extensions.embedding import TabPFNEmbedding as Emb1
    X = np.random.rand(50, 10).astype(np.float32)
    y = np.random.randint(0, 2, 50).astype(np.float32)
    clf1 = Clf1(n_estimators=1, device='cpu')
    clf1.fit(X, y)
    emb1 = Emb1(tabpfn_clf=clf1, n_fold=5)
    emb1.fit(X, y)
    out1 = emb1.get_embeddings(X, y, X[:10], data_source='test')
    print(f'  Shape: {out1.shape}, Embed dim: {out1.shape[-1]}')
except Exception as e:
    print(f'  FAILED: {e}')

print()
print('=' * 60)
print('TEST 2: with model_path=tabpfn-v3-classifier-v3_20260417_binary.ckpt')
print('=' * 60)
try:
    clf2 = Clf1(n_estimators=1, device='cpu', model_path='tabpfn-v3-classifier-v3_20260417_binary.ckpt')
    clf2.fit(X, y)
    emb2 = Emb1(tabpfn_clf=clf2, n_fold=5)
    emb2.fit(X, y)
    out2 = emb2.get_embeddings(X, y, X[:10], data_source='test')
    print(f'  Shape: {out2.shape}, Embed dim: {out2.shape[-1]}')
except Exception as e:
    print(f'  FAILED: {e}')

print()
print('=' * 60)
print('TEST 3: tabpfn.TabPFNClassifier (no model_path)')
print('=' * 60)
try:
    from tabpfn import TabPFNClassifier as Clf3
    clf3 = Clf3(n_estimators=1, device='cpu')
    clf3.fit(X, y)
    emb3 = Emb1(tabpfn_clf=clf3, n_fold=5)
    emb3.fit(X, y)
    out3 = emb3.get_embeddings(X, y, X[:10], data_source='test')
    print(f'  Shape: {out3.shape}, Embed dim: {out3.shape[-1]}')
except Exception as e:
    print(f'  FAILED: {e}')

print()
print('=' * 60)
print('TEST 4: n_fold=0 vanilla embeddings')
print('=' * 60)
try:
    clf4 = Clf1(n_estimators=1, device='cpu')
    clf4.fit(X, y)
    emb4 = Emb1(tabpfn_clf=clf4, n_fold=0)
    emb4.fit(X, y)
    out4 = emb4.get_embeddings(X, y, X[:10], data_source='test')
    print(f'  Shape: {out4.shape}, Embed dim: {out4.shape[-1]}')
except Exception as e:
    print(f'  FAILED: {e}')
