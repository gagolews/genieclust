options(encoding="UTF-8")
# ############################################################################ #
#   Marek's R graphics package style hacks                                     #
#   aka "you don't need ggplot2 to look cool"                                  #
#                                                                              #
#   Don't try this at home, kids!!!                                            #
#                                                                              #
#   Copyleft (C) 2020, Marek Gagolewski <https://www.gagolewski.com>           #
#                                                                              #
#                                                                              #
#   This program is free software: you can redistribute it and/or modify       #
#   it under the terms of the GNU Affero General Public License                #
#   Version 3, 19 November 2007, published by the Free Software Foundation.    #
#   This program is distributed in the hope that it will be useful,            #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the               #
#   GNU Affero General Public License Version 3 for more details.              #
#   You should have received a copy of the License along with this program.    #
#   If this is not the case, refer to <https://www.gnu.org/licenses/>.         #
#                                                                              #
# ############################################################################ #


library("Cairo")
# CairoFonts(
#     regular="Ubuntu Condensed:style=Regular",
#     bold="Ubuntu:style=Medium",
#     italic="Ubuntu:style=Light Italic",
#     bolditalic="Ubuntu:style=Medium Italic",
#     symbol="Ubuntu Condensed"
# )
CairoFonts(
    regular="Alegreya Sans:style=Regular",
    italic="Alegreya Sans:style=Italic",
    bold="Alegreya Sans:style=Medium",
    bolditalic="Alegreya Sans:style=Medium Italic",
    symbol="TeX Gyre Pagella:style=Regular"
)

setHook("before.plot.new", function() {
    if (all(par("mar") == c(5.1, 4.1, 4.1, 2.1))) {
        # the above is the default `mar`, let's change it to a new default!
        par(mar=c(2.5,2.5,1,0.5))
    }
#     if (..output_language!="tex") {
#         par(family="Ubuntu")
#     }
    par(tcl=-0.25)
    par(mgp=c(1.25, 0.5, 0))
    par(cex.main=1)
    par(font.main=2)
    par(cex.axis=0.9)
    par(cex.lab=1)
    par(font.lab=3)
}, "replace")



plot.window_new <- function (xlim, ylim, log = "", asp = NA, ...)
{
    .External.graphics(C_plot_window, xlim, ylim, log, asp, ...)

    if (par("ann") != FALSE) {
        x1 <- par("usr")[1]
        x2 <- par("usr")[2]
        if (par("xlog")) { x1 <- 10^x1; x2 <- 10^x2 }
        y1 <- par("usr")[3]
        y2 <- par("usr")[4]
        if (par("ylog")) { y1 <- 10^y1; y2 <- 10^y2 }
        rect(x1, y1, x2, y2, col="#00000010")
        abline(v=axTicks(1), col="white", lwd=1.5, lty=1)
        abline(h=axTicks(2), col="white", lwd=1.5, lty=1)
        box()
    }
    invisible()
}

environment(plot.window_new) <- environment(plot.window)
unlockBinding("plot.window", getNamespace("graphics"))
assign("plot.window", plot.window_new, getNamespace("graphics"))

axis_new <- function (side, at = NULL, labels = TRUE, tick = TRUE, line = -0.25,
          pos = NA, outer = FALSE, font = NA, lty = "solid", lwd = 0,
          lwd.ticks = 1, col = NULL, col.ticks = NULL, hadj = NA,
          padj = NA, gap.axis = NA, ...)
{
    if (is.null(col) && !missing(...) && !is.null(fg <- list(...)$fg))
        col <- fg
    invisible(.External.graphics(C_axis, side, at, as.graphicsAnnot(labels),
                                 tick, line, pos, outer, font, lty, lwd,
                                 lwd.ticks, col, col.ticks, hadj, padj,
                                 gap.axis, ...))
}
environment(axis_new) <- environment(axis)
unlockBinding("axis", getNamespace("graphics"))
assign("axis", axis_new, getNamespace("graphics"))

################################################################################
# Marek's R graphics package style hacks                                  EOF. #
################################################################################
