`genieclust` package CHANGELOG
==============================


## genieclust 0.XX (under development)

-   The full distance matrix is no longer required for computing an
    exact MST -- the distances might be computed on the fly; this is
    currently supported for `"euclidean"`,
    `"cityblock"`, and `"cosine"`
    distances. The upgrade saves a lot of memory ($O(n)$ instead
    of $O(n^2)$)  -- `genieclust` can solve much larger problems now.

-   TODO: see a list of resolved github issues....

-   TODO: GIc...

-   TODO: `sklearn.neighbors.NearestNeighbors`,
    `faiss` or other search data structures are used to
    determine few first NNs when computing the mutual reachability
    distance a.k.a. the "core distance".

-   [DEPRECATED] `internal.core_distance` and
    `internal.merge_boundary_points` are now available via
    `deprecated.*`.

-   [INTERNAL] Most of the code was rewritten in C++, so that:

    a.  they can be used in other projects (in particular the `IntDict`,
        `DisjointSets` and `GiniDisjointSets` classes),

    b.  genieclust can be made available for
        other environments in the future.

-   [INTERNAL] Use OpenMP for distance computations.

-   [BUGFIX] Internal function `MST_pair()` did not return all
    weights of the MST edges.


## genieclust 0.1a3 (unreleased)

-   The `Genie` class constructor has a new parameter:
    `postprocess`, which allows for merging boundary
    points with their nearest core points (in effect for smoothing
    parameters M>1).


## genieclust 0.1a2 (2018-05-23)

-   Initial PyPI release.
