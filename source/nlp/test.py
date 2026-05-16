from ..data.queries import return_chunks
from ..data import partition_indices

data = return_chunks(200)
idx = partition_indices(data,1000)
for l, r in idx:
    print(data.loc[l:r,:])
