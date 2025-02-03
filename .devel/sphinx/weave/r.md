



# R Interface Examples

The latest stable release of the R package `genieclust` is available from the
[CRAN](https://cran.r-project.org/web/packages/genieclust/) repository.
We can install it by calling:


``` r
install.packages("genieclust")
```


Below are a few basic examples of how to interact with the package.


``` r
library("genieclust")
```



Let's consider the [Sustainable Society Indices](http://www.ssfindex.com/)
dataset that measures the Human, Environmental, and Economic Wellbeing
in each country. There are seven categories on the scale $[0, 10]$.


``` r
# see https://github.com/gagolews/genieclust/tree/master/devel/sphinx/weave
ssi <- read.csv("ssi_2016_categories.csv", comment.char="#")
X <- as.matrix(ssi[,-1])    # everything except the Country (1st) column
dimnames(X)[[1]] <- ssi[,1] # set row names
head(X)  # preview
##           BasicNeeds PersonalDevelopmentAndHealth WellBalancedSociety
## Albania       9.6058                       7.9596              6.9926
## Algeria       9.0212                       7.3365              4.2039
## Angola        5.9728                       5.6928              2.1401
## Argentina     9.8320                       8.3506              3.8952
## Armenia       9.4469                       7.4205              6.2892
## Australia    10.0000                       8.5909              6.1055
##           NaturalResources ClimateAndEnergy Transition Economy
## Albania             6.6343           4.6217     2.1025  3.0565
## Algeria             5.2772           2.6627     3.0741  6.1543
## Angola              6.7594           6.2122     1.8988  3.7535
## Argentina           5.4535           3.3003     6.3899  5.3406
## Armenia             6.4363           2.8543     2.4342  3.8296
## Australia           4.1307           1.6278     7.5395  7.5931
```


The `genieclust` interface is compatible with R's workhorse
for hierarchical clustering, `stats::hclust()`.
Yet, for efficiency reasons, it is better to feed `genieclust::gclust()`
with the input matrix directly:



``` r
# faster than gclust(dist(X)):
h <- gclust(X)  # default: gini_threshold=0.3, distance="euclidean"
print(h)
## 
## Call:
## gclust.mst(d = mst.default(d, M = 1L, distance = distance, verbose = verbose,     cast_float32 = cast_float32), gini_threshold = gini_threshold,     verbose = verbose)
## 
## Cluster method   : Genie(0.3) 
## Distance         : euclidean 
## Number of objects: 154
```

In order to extract a desired *k*-partition, we can call `stats::cutree()`:


``` r
y_pred <- cutree(h, k=3)
sample(y_pred, 25)  # preview
##                Iran            Slovenia             Morocco 
##                   1                   3                   1 
## Trinidad and Tobago        Sierra Leone            Colombia 
##                   1                   2                   1 
##        Turkmenistan               Syria             Nigeria 
##                   1                   1                   2 
##             Tunisia               Ghana              Sweden 
##                   1                   2                   1 
##         Netherlands              Bhutan              Mexico 
##                   3                   1                   1 
##             Jamaica        Saudi Arabia             Hungary 
##                   1                   1                   3 
##                Peru          Kazakhstan               Spain 
##                   1                   1                   3 
##               Japan        Korea, North               China 
##                   1                   1                   1 
##             Liberia 
##                   2
```

This gives the cluster IDs allocated to each country.
Let's depict the obtained partition using the `rworldmap` package:


``` r
library("rworldmap")  # see the package's manual for details
mapdata <- data.frame(Country=dimnames(X)[[1]], Cluster=y_pred)
mapdata <- joinCountryData2Map(mapdata, joinCode="NAME", nameJoinColumn="Country")
mapCountryData(mapdata, nameColumnToPlot="Cluster", catMethod="categorical",
    missingCountryCol="white", colourPalette=palette.colors(3, "R4"),
    mapTitle="")
```

(fig:ssi-map)=
```{figure} r-figures/ssi-map-1.*
Countries grouped w.r.t. the SSI categories.
```


We can compute, e.g., the average indicators in each identified group:


``` r
t(aggregate(as.data.frame(X), list(Cluster=y_pred), mean))[-1, ]
##                                [,1]   [,2]   [,3]
## BasicNeeds                   9.0679 5.2689 9.8178
## PersonalDevelopmentAndHealth 7.5081 5.9312 8.2995
## WellBalancedSociety          4.8869 2.8682 6.8272
## NaturalResources             5.6633 7.0040 6.3743
## ClimateAndEnergy             3.6241 7.0818 3.5947
## Transition                   4.0749 2.6300 7.3402
## Economy                      5.5127 3.5411 4.2742
```


Dendrogram plotting is also possible.
For greater readability, we will restrict ourselves to a smaller sample;
namely, to the 37 members of the [OECD](https://en.wikipedia.org/wiki/OECD):


``` r
oecd <- c("Australia", "Austria", "Belgium", "Canada", "Chile", "Colombia",
"Czech Republic", "Denmark", "Estonia", "Finland", "France", "Germany",
"Greece", "Hungary", "Iceland", "Ireland", "Israel", "Italy", "Japan",
"Korea, South", "Latvia", "Lithuania", "Luxembourg", "Mexico", "Netherlands",
"New Zealand", "Norway", "Poland", "Portugal", "Slovak Republic", "Slovenia",
"Spain", "Sweden", "Switzerland", "Turkey", "United Kingdom", "United States")
X_oecd <- X[dimnames(X)[[1]] %in% oecd, ]
```



``` r
h_oecd <- gclust(X_oecd)
plot(as.dendrogram(h_oecd), horiz=TRUE)
```

(fig:ssi-oecd-dendrogram)=
```{figure} r-figures/ssi-oecd-dendrogram-1.*
Cluster dendrogram for the OECD countries.
```



Conclusion:

* If we are only interested in a specific partition,
calling `genie()` directly will be slightly faster than referring to
`cutree(gclust(...))`.

* `genieclust` also features partition similarity scores
(such as the adjusted Rand index) that can be used as
external cluster validity measures.

For more details, refer to the package's {any}`documentation <../rapi>`.
Also, see the Python examples regarding noise points detection,
benchmarking, timing, etc.

*To learn more about R, check out Marek's recent open-access (free!) textbook*
[Deep R Programming](https://deepr.gagolewski.com/)
{cite}`deepr`.
