# Plotting the convex hull of a point set:

import scipy.spatial
np.random.seed(123)
X = np.random.randn(100, 2)
hull = scipy.spatial.ConvexHull(X)
genieclust.plots.plot_scatter(X)
genieclust.plots.plot_segments(hull.simplices, X, style="r--")
plt.show()                                 # doctest: +SKIP
