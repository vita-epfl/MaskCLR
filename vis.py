from sklearn.datasets import load_digits
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
#from tsnecuda import TSNE

digits = load_digits()

print(type(digits.data), digits.data.shape) # <class 'numpy.ndarray'> (1797, 64)
print(type(digits.target), digits.target.shape) # <class 'numpy.ndarray'> (1797,)

embeddings = TSNE(n_jobs=4).fit_transform(digits.data)


#embeddings = TSNE().fit_transform(digits.data)
vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]
plt.figure(figsize=(14, 10))
plt.scatter(vis_x, vis_y, c=digits.target, cmap=plt.cm.get_cmap("jet", 10), marker='.')
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()

save_path = 'ls2.png'
plt.savefig(save_path, bbox_inches="tight")