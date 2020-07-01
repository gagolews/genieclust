# Here are the first 10 plotting styles:

ncol = len(genieclust.plots.col)
nmrk = len(genieclust.plots.mrk)
mrk_recycled = np.tile(
    genieclust.plots.mrk,
    int(np.ceil(ncol/nmrk)))[:ncol]
for i in range(10):
    plt.text(i, 0, i, horizontalalignment="center")
    plt.plot(
        i, 1, marker=mrk_recycled[i],
        color=genieclust.plots.col[i],
        markersize=25)
plt.title("Plotting styles for labels=0,1,...,9")
plt.ylim(-3,4)
plt.axis("off")
plt.show()
