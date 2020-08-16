"""
genieclust Python Package
"""


# ############################################################################ #
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


from . import plots
from . import inequity
from . import tools
from . import compare_partitions
from . import internal
from .genie import Genie, GIc

__version__ = '0.9.5.dev0'  # see also ../setup.py
