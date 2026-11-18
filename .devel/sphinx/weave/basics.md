



# Basic Usage (Python)

*Genie* {cite}`genieins` is an agglomerative hierarchical clustering
algorithm. The idea behind *Genie* is beautifully simple. First, it makes each
individual point the sole member of its own cluster. Then, it keeps merging pairs
of the closest clusters, one after another. However, to **prevent
the formation of clusters of highly imbalanced sizes**, a point group of the
smallest size will sometimes be combined with its nearest counterpart.

In the following sections, we demonstrate that Genie often outperforms
other popular methods in terms of clustering [quality](benchmarks_ar)
and [speed](timings).

Here are a few examples of basic interactions with the Python version
of the **genieclust** {cite}`genieclust` package,
which we can install from [PyPI](https://pypi.org/project/genieclust/), e.g.,
via a call to `pip3 install genieclust` from the command line.


::::{note}
For best performance, advanced users will benefit from compiling the package
from sources:

```bash
CPPFLAGS="-O3 -march=native" pip3 install quitefastmst deadwood lumbermark --force --no-binary="quitefastmst,deadwood,genieclust"
```

🚧 TO DO (help needed): How to enable OpenMP support on macOS in **quitefastmst**?
::::



``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import genieclust
import deadwood

plot_params = dict(s=10, markers="o", alpha=0.75, asp=1)
```


































