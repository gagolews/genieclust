# An example scatter plots where each point is assigned one of
# two distinct labels:

n = np.r_[100, 50]
X = np.r_[np.random.randn(n[0], 2), np.random.randn(n[1], 2)+2.0]
l = np.repeat([0, 1], n)
genieclust.plots.plot_scatter(X, labels=l)
plt.show()                                   # doctest: +SKIP
