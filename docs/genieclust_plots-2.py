# Here are the first 10 plotting styles:

ncol = len(genieclust.plots.col)
nmrk = len(genieclust.plots.mrk)
mrk_recycled = np.tile(
    genieclust.plots.mrk,
    int(np.ceil(ncol/nmrk)))[:ncol]
for i in range(10):                               # doctest: +SKIP
    plt.text(                                     # doctest: +SKIP
        i, 0, i, horizontalalignment="center")    # doctest: +SKIP
    plt.plot(                                     # doctest: +SKIP
        i, 1, marker=mrk_recycled[i],             # doctest: +SKIP
        color=genieclust.plots.col[i],            # doctest: +SKIP
        markersize=25)                            # doctest: +SKIP
plt.title("Plotting styles for labels=0,1,...,9") # doctest: +SKIP
plt.ylim(-3,4)                                    # doctest: +SKIP
plt.axis("off")                                   # doctest: +SKIP
plt.show()                                        # doctest: +SKIP
