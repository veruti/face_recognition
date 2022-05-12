import numpy as np

d = 64  # dimension
nb = 100000  # database size
nq = 10000  # nb of queries

np.random.seed(1234)  # make reproducible
xb = np.random.random((nb, d)).astype("float32")
xb[:, 0] += np.arange(nb) / 1000.0
xq = np.random.random((nq, d)).astype("float32")
xq[:, 0] += np.arange(nq) / 1000.0

print(f"xb shape: {xb.shape}")

import faiss  # make faiss available

index = faiss.IndexFlatIP(d)  # build the index

print(xb.shape)
print(faiss.normalize_L2(xb))
index.add(xb)  # add vectors to the index
print(index.ntotal)

faiss.normalize_L2(xb)

# k = 4  # we want to see 4 nearest neighbors
D = index.search(xq[:5], 1)  # sanity check
print(D)
# print(I)
# print(D)
# D, I = index.search(xq, k)  # actual search
# print(I[:5])  # neighbors of the 5 first queries
# print(I[-5:])  # neighbors of the 5 last queries
