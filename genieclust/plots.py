"""
Various plotting functions DEPRECATED

DEPRECATED: functions moved to 'deadwood'
"""

# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2020-2026, Marek Gagolewski <https://www.gagolewski.com>      #
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


import deadwood
import warnings


def plot_scatter(
        *args,
        **kwargs
    ):
    """
    DEPRECATED: Use ``deadwood.plot_scatter`` instead
    """
    warnings.warn("This function is DEPRECATED; use deadwood.plot_scatter instead", DeprecationWarning)
    return deadwood.plot_scatter(*args, **kwargs)


def plot_segments(*args, **kwargs):
    """
    DEPRECATED: Use ``deadwood.plot_segments`` instead
    """
    warnings.warn("This function is DEPRECATED; use deadwood.plot_segments instead", DeprecationWarning)
    return deadwood.plot_segments(*args, **kwargs)
