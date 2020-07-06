# Copyright (C) 2020, Marek Gagolewski, https://www.gagolewski.com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.




# We suggested that "parametric" datasets g2mg, h2mg should be studied separately.
# Subset: not g2mg, not h2mg
res2 = res.loc[~res.battery.isin(["g2mg", "h2mg"]), :]

# Subset: [large datasets]
res2 = res2.loc[res2.dataset.isin(dims.dataset[dims.n>10_000]), :]

res2["dataset"] = res2["battery"] + "/" + res2["dataset"]



# For each dataset, method, compute maximal scores across
# all available true (reference) labels (if there alternative labellings
# of a given dataset):
res_max = res2.groupby(["dataset", "method"]).max().\
    reset_index().drop(["labels"], axis=1)
#res_max.head()




# which similarity measure to report below:
__order=res_max.groupby("method")[similarity_measure].mean().sort_values().index

plt.rcParams["figure.figsize"] = (8,8)
sns.boxplot(y="method", x=similarity_measure, data=res_max,
            order=__order,
            orient="h",
            showmeans=True,
            meanprops=dict(markeredgecolor="k", marker="x"))
plt.show()

res_max.groupby("method")[similarity_measure].describe().T.round(3)


res_max.set_index(["dataset", "method"])[similarity_measure].unstack().round(3)

