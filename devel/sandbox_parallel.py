import numpy as np
from genieclust.genie import *
from genieclust.inequity import *
from genieclust.compare_partitions import *
import time
import gc, os


import scipy.spatial.distance
from rpy2.robjects.packages import importr
stats = importr("stats")
genie = importr("genie")
import numpy as np
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()



np.random.seed(123)
n = 50_000
d = 69
X = np.random.normal(size=(n,d))
labels = np.random.choice(np.r_[1,2,3,4,5,6,7,8], n)

k = len(np.unique(labels[labels>=0]))
# center X + scale (NOT: standardize!)
X = (X-X.mean(axis=0))/X.std(axis=None, ddof=1)
g = 0.3
metric="euclidean"


print("n=%d, d=%d, g=%.2f, k=%d" %(n,d,g,k))
print("OMP_NUM_THREADS=%d"%int(os.environ["OMP_NUM_THREADS"]))

t01 = time.time()
res1 = Genie(k, g, exact=True, affinity=metric).fit_predict(X)+1
t11 = time.time()
print("t_py =%.3f" % (t11-t01))

assert len(np.unique(res1)) == k

t02 = time.time()
res2 = stats.cutree(genie.hclust2(objects=X, d=metric, thresholdGini=g), k)
t12 = time.time()
print("t_r  =%.3f (rel_to_py=%.3f)" % (t12-t02,(t02-t12)/(t01-t11)))

res2 = np.array(res2, np.intp)
assert len(np.unique(res2)) == k
ari = adjusted_rand_score(res1, res2)
assert ari>1.0-1e-12


t03 = time.time()
res3 = Genie(k, g, exact=False, compute_full_tree=False, affinity=metric).fit_predict(X)+1
t13 = time.time()
ari = adjusted_rand_score(res1, res3)
print("t_py2=%.3f (rel_to_py=%.3f; ari=%.8f)" % (t13-t03,(t03-t13)/(t01-t11), ari))
assert ari>1.0-1e-6



# 2020-04-14:
#n=50000, d=69, g=0.30, k=8
#OMP_NUM_THREADS=8
#t_py =18.351 [mostly because we use float32]
#t_r  =42.293 (rel_to_py=2.305)
#t_py2=11.611 (rel_to_py=0.633; ari=1.00000000)

# 2020-04-14:
#n=50000, d=69, g=0.30, k=8
#OMP_NUM_THREADS=4
#t_py =21.677
#t_r  =44.671 (rel_to_py=2.061)
#t_py2=12.770 (rel_to_py=0.589; ari=1.00000000)
