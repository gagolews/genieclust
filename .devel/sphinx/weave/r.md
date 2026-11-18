



# Basic Usage (R)

The latest stable release of the R package **genieclust** is available from the
[CRAN](https://CRAN.R-project.org/package=genieclust) repository.
Install it by calling:


``` r
install.packages("genieclust")
```


::::{note}
For best performance, advanced users will benefit from compiling the package
from sources:

```r
Sys.setenv(CXX_DEFS="-O3 -march=native")  # for gcc and clang
install.packages(c("quitefastmst", "deadwood", "genieclust"), type="source")
```

To enable OpenMP on macOS, execute `brew install libomp`, and then call in R:

```r
Sys.setenv(CXX_DEFS="-O3 -march=native -Xpreprocessor -fopenmp")
install.packages(c("quitefastmst", "deadwood", "genieclust"), type="source")
```
::::



Below are a few basic examples of interacting with the package.


``` r
library("genieclust")
```


Let's take the [Sustainable Society Indices](http://www.ssfindex.com/)
dataset that measures the Human, Environmental, and Economic Wellbeing
in each country. There are seven categories on the scale $[0, 10]$.


``` r
# see https://github.com/gagolews/genieclust/tree/master/devel/sphinx/weave
ssi <- read.csv("ssi_2016_categories.csv", comment.char="#")
X <- as.matrix(ssi[,-1])    # everything except the Country (1st) column
dimnames(X)[[1]] <- ssi[,1] # set row names
head(X)  # preview
```

```
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


The API of **genieclust** is compatible with R's workhorse
for hierarchical clustering, `stats::hclust()`, which accepts
a complete pairwise distance matrix.  Yet, for better performance,
it is better to feed `genieclust::gclust()` with the input data matrix
directly:



``` r
# faster than gclust(dist(X)):
h <- gclust(X)  # default: gini_threshold=0.3, distance="euclidean"
print(h)
```

```
## 
## Call:
## gclust.mst(d = mst(d, M = M, distance = distance, verbose = verbose,     ...), gini_threshold = gini_threshold, verbose = verbose)
## 
## Cluster method   : Genie(0.3) 
## Distance         : euclidean 
## Number of objects: 154
```

In order to extract a desired *k*-partition, we can call `stats::cutree()`:


``` r
y_pred <- cutree(h, k=3)  # also: genie(X, 3)
sample(y_pred, 25)  # preview
```

```
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


::::{note}
If we are only interested in the partition of a specific cardinality,
calling `genie()` directly will be slightly faster than referring to
`cutree(gclust(...))`.
::::

Let's depict the obtained partition using the **rworldmap** package:









