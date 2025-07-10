"""
genieclust Python Package
"""


# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2020-2025, Marek Gagolewski <https://www.gagolewski.com>      #
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

# version string, e.g., "1.0.0.9001" or "1.1.1"
__version__ = "1.1.99.9666"


from . import plots
from . import inequality
from . import tools
from . import compare_partitions
from . import cluster_validity
from . import internal
from . import fastmst
from . import oldmst
from .genie import Genie, GIc


# TODO: Why the OMP_NUM_THREADS envvar is not honoured automatically?
import os
_omp_max_threads_original = fastmst.omp_get_max_threads()
_omp_num_threads_envvar = int(os.getenv("OMP_NUM_THREADS", -1))
if _omp_num_threads_envvar > 0:
    fastmst.omp_set_num_threads(_omp_num_threads_envvar)
