# Plotting the minimum spanning tree:

np.random.seed(123)
X = np.random.randn(100, 2)
mst = genieclust.internal.mst_from_distance(X, "euclidean")
genieclust.plots.plot_scatter(X)
genieclust.plots.plot_segments(mst[1], X, style="m-.")
plt.axis("equal")                          # doctest: +SKIP
plt.show()                                 # doctest: +SKIP
