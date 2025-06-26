import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
plt.style.use("seaborn-v0_8")  # plot style template
_colours = [  # the "R4" palette from R
    "#000000f0", "#DF536Bf0", "#61D04Ff0", "#2297E6f0",
    "#28E2E5f0", "#CD0BBCf0", "#F5C710f0", "#999999f0"
]
_linestyles = [
    "solid", "dashed", "dashdot", "dotted"
]
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    # each plotted line will have a different plotting style
    color=_colours, linestyle=_linestyles*2
)
plt.rcParams["patch.facecolor"] = _colours[0]


fname = "~/Python/genieclust/.devel/perf_knn_202506-apollo.csv"
res = pd.read_csv(fname, comment="#")
# assert np.all(res.groupby(["method", "n", "d", "k", "nthreads"])["elapsed"].count() == 3)
print("Max Δdist = %g" % (res.loc[:, "Δdist"].max(), ))
print("Max Δidx  = %g" % (res.loc[:, "Δidx"].max(), ))
res.drop(columns=["Δdist", "Σdist", "Δidx", "time", "host"], inplace=True)

res_method = res.method.astype("category")
res_method = res_method.cat.rename_categories(dict(
    r_genieclust_kdtree="genieclust_kdtree",
))
res["method"] = res_method.astype("str")




# min of three trials - elapsed time in seconds
times = res.groupby(["method", "n", "d", "k", "nthreads"])["elapsed"].min().rename("time").reset_index()
times.head()

print(res.query("nthreads==1").groupby(["n", "d", "k"]).count().to_markdown())


# r_vs_py = times.set_index(["method"]).loc[["genieclust_kdtree", "py_genieclust_kdtree"], :].reset_index().set_index(["n", "d", "k", "nthreads", "method"]).unstack()
# print(((r_vs_py.iloc[:, 0] - r_vs_py.iloc[:, 1])/r_vs_py.iloc[:, 0]).describe())


times.query("n == 1208592 and nthreads == 1 and k == 1").set_index(["method", "d"]).time.unstack().sort_values(0)


# plt.clf()
# sns.lineplot(x="n", y="time", hue="d", style="method", data=times, palette=_colours)
# # plt.yscale("log")
# # plt.xscale("log")
# plt.show()
#
# stop()


plt.clf()
q, x = [
    ("d == 2 and nthreads == 1 and k == 10 and method != 'r_genieclust_brute'", "n"),
    ("n == 2**18 and nthreads == 1 and k == 10", "d")
][1]
sns.lineplot(x=x, y="time", hue="method", style="method", data=times.query(q), palette=_colours)
plt.title(q)
#plt.yscale("log")
#plt.xscale("log")
plt.show()
