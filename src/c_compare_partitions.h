/*  External Cluster Validity Measures
 *
 *  Adjusted- and Nonadjusted Rand Score,
 *  Adjusted- and Nonadjusted Fowlkes-Mallows Score,
 *  Adjusted-, Normalised and Nonadjusted Mutual Information Score,
 *  Normalised Accuracy, Pair Sets Index
 *  (for vectors of "small" ints)
 *
 *
 *  References
 *  ==========
 *
 *  Hubert L., Arabie P., Comparing Partitions,
 *  Journal of Classification 2(1), 1985, pp. 193-218, esp. Eqs. (2) and (4)
 *
 *  Vinh N.X., Epps J., Bailey J.,
 *  Information theoretic measures for clusterings comparison:
 *  Variants, properties, normalization and correction for chance,
 *  Journal of Machine Learning Research 11, 2010, pp. 2837-2854.
 *
 *  Rezaei M., Franti P., Set matching measures for external cluster validity,
 *  IEEE Transactions on Knowledge and Data Mining 28(8), 2016, pp. 2173-2186,
 *  doi:10.1109/TKDE.2016.2551240
 *
 *
 *  Copyleft (C) 2018-2025, Marek Gagolewski <https://www.gagolewski.com>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Affero General Public License
 *  Version 3, 19 November 2007, published by the Free Software Foundation.
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU Affero General Public License Version 3 for more details.
 *  You should have received a copy of the License along with this program.
 *  If this is not the case, refer to <https://www.gnu.org/licenses/>.
 */


#ifndef __c_compare_partitions_h
#define __c_compare_partitions_h

#include "c_common.h"
#include "c_scipy_rectangular_lsap.h"
#include <algorithm>
#include <cmath>
#include <vector>


/*! (t choose 2)
 *
 * @param t
 * @return t*(t-1.0)*0.5
 */
inline double Ccomb2(double t)
{
    return t*(t-1.0)*0.5;
}


/*! Computes both the minimum and the maximum in an array
 *
 * @param x c_contiguous array of length n
 * @param n length of x
 * @param xmin [out] the minimum of x
 * @param xmax [out] the maximum of x
 */
template<class T>
void Cminmax(const T* x, Py_ssize_t n, T* xmin, T* xmax)
{
    *xmin = x[0];
    *xmax = x[0];

    for (Py_ssize_t i=1; i<n; ++i) {
        if      (x[i] < *xmin) *xmin = x[i];
        else if (x[i] > *xmax) *xmax = x[i];
    }
}



/*!
 * Stores AR and FM scores as well as their adjusted/normalised versions.
 */
struct CComparePartitionsPairsResult {
    double ar;
    double r;
    double fm;
    double afm;
};


/*!
 * Stores mutual information-based scores
 */
struct CComparePartitionsInfoResult {
    double mi;
    double nmi;
    double ami;
};



/*!
 * Stores normalised set-matching scores
 */
struct CCompareSetMatchingResult {
    double psi_unclipped;
    double spsi_unclipped;
};




/*! Normalising permutation for the columns of a confusion matrix
 *
 *  Determines the reordering of columns in a given confusion matrix
 *  so that the sum of the elements on the main diagonal is the largest
 *  possible (by solving the maximal assignment problem).
 *
 *  Comes in handy if C summarises the results generated
 *  by clustering algorithms, where actual label values do not matter
 *  (e.g., (1, 2, 0) can be remapped to (0, 2, 1) with no change in meaning.
 *
 *
 *  @param C a c_contiguous confusion matrix of size xc*yc
 *  @param xc number of rows in C
 *  @param yc number of columns in C; xc <= yc
 *  @param Iout [out] output sequence of length yc
 *
 *  Note that Iout is modified in-place (overwritten).
 */
template<class T1, class T2>
void Cnormalizing_permutation(
    const T1* C, Py_ssize_t xc, Py_ssize_t yc, T2* Iout
) {
    GENIECLUST_ASSERT(xc <= yc);

    std::vector<bool> column_used(yc, false);

    Py_ssize_t retval = linear_sum_assignment(
        C, xc, yc, Iout, /*minimise*/false
    );
    GENIECLUST_ASSERT(retval == 0);

    // only Iout[0]..Iout[xc-1] are set
    Py_ssize_t i;
    for (i=0; i<xc; ++i) {
        column_used[ Iout[i] ] = true;
    }

    // the remainder:
    for (Py_ssize_t k=0; k<yc; ++k) {
        if (!column_used[k]) {
            column_used[k] = true;
            Iout[i] = k;
            i++;

            if (i == yc) break;
        }
    }
}


/*! Applies pivoting to a given confusion matrix
 *
 *  Permutes the rows and columns so that the sum of the elements
 *  on the main diagonal is the largest possible (by solving
 *  the maximal assignment problem).
 *
 *  See Cnormalizing_permutation().
 *
 *
 *  @param C a c_contiguous confusion matrix of size xc*yc
 *  @param xc number of rows in C
 *  @param yc number of columns in C; xc <= yc
 *  @param Cout [out] output matrix after pivoting
 *
 *  Note that Cout is modified in-place (overwritten).
 */
template<class T>
void Capply_pivoting(
    const T* C, Py_ssize_t xc, Py_ssize_t yc, T* Cout/*, bool use_sum=false*/
) {
    GENIECLUST_ASSERT(xc <= yc);

//     if (use_sum) {

    std::vector<Py_ssize_t> output_col4row(yc);

    Cnormalizing_permutation(C, xc, yc, /*retval*/output_col4row.data());

    Py_ssize_t i;
    for (i=0; i<yc; ++i) {
        for (Py_ssize_t j=0; j<xc; ++j)
            Cout[yc*j+i] = C[yc*j+output_col4row[i]];
    }

//     }
//     WARNING: not tested yet
//     else { // use max
//         for (Py_ssize_t ij=0; ij<xc*yc; ++ij)
//             Cout[ij] = C[ij];
//
//         for (Py_ssize_t i=0; i<xc-1; ++i) {
//             Py_ssize_t wi = i, wj = i;
//
//             for (Py_ssize_t ni=i; ni<xc; ++ni) {
//                 for (Py_ssize_t nj=i; nj<yc; ++nj) {
//                 // find wi, wj = argmax C[ni,nj]
//                 if (C[wi*yc+wj] < C[ni*yc+nj]) { wi = ni; wj = nj; }
//             }
//             // swap columns i and wj
//             for (Py_ssize_t j=0; j<xc; ++j) {
//                 std::swap(C[j*yc+i], C[j*yc+wj]);
//             }
//             // swap rows i and wi
//             for (Py_ssize_t k=0; k<yc; ++k) {
//                 std::swap(C[i*yc+k], C[wi*yc+k]);
//             }
//         }
//     }
}



/*! Computes the confusion matrix (as a dense matrix) - a 2-way contingency table
 *
 * @param C [out] a c_contiguous matrix of size xc*yc
 *      where C[i-xmin,j-ymin] is the number of k such that x[k]==i,y[k]==j;
 * @param xc number of rows in C
 * @param yc number of columns in C
 * @param xmin the minimum of x
 * @param ymin the minimum of y
 * @param x,y c_contiguous arrays of length n with x[i], y[i] being integers
 * in [xmin, xmin+xc) and [ymin, ymin+yc), respectively,
 * denoting the class/cluster of the i-th observation
 * @param n length of the two arrays
 *
 * The elements in C are modified in-place.
 */
template<class O, class T>
void Ccontingency_table(O* Cout, Py_ssize_t xc, Py_ssize_t yc,
        T xmin, T ymin,
        const T* x, const T* y, Py_ssize_t n)
{
    for (Py_ssize_t j=0; j<xc*yc; ++j)
        Cout[j] = 0;

    for (Py_ssize_t i=0; i<n; ++i) {
        GENIECLUST_ASSERT(   0 <= (x[i]-xmin)*yc +(y[i]-ymin));
        GENIECLUST_ASSERT(xc*yc > (x[i]-xmin)*yc +(y[i]-ymin));
        Cout[(x[i]-xmin)*yc +(y[i]-ymin)] += 1;
    }
}


/*! Computes the adjusted and nonadjusted Rand- and FM scores
 *  based on a given confusion matrix.
 *
 *  References
 *  ==========
 *
 *  Hubert, L., Arabie, P., Comparing Partitions,
 *  Journal of Classification 2(1), 1985, pp. 193-218, esp. Eqs. (2) and (4)
 *
 *  @param C a c_contiguous confusion matrix of size xc*yc
 *  @param xc number of rows in C
 *  @param yc number of columns in C
 *
 *  @return the computed scores
 */
template<class T>
CComparePartitionsPairsResult Ccompare_partitions_pairs(const T* C,
    Py_ssize_t xc, Py_ssize_t yc)
{
    double n = 0.0; // total sum (length of the underlying x and y = number of points)
    for (Py_ssize_t ij=0; ij<xc*yc; ++ij)
        n += C[ij];

    double sum_comb_x = 0.0, sum_comb_y = 0.0, sum_comb = 0.0;

    for (Py_ssize_t i=0; i<xc; ++i) {
        double t = 0.0;
        for (Py_ssize_t j=0; j<yc; ++j) {
            t += C[i*yc+j];
            sum_comb += Ccomb2(C[i*yc+j]);
        }
        sum_comb_x += Ccomb2(t);
    }

    for (Py_ssize_t j=0; j<yc; ++j) {
        double t = 0.0;
        for (Py_ssize_t i=0; i<xc; ++i) {
            t += C[i*yc+j];
        }
        sum_comb_y += Ccomb2(t);
    }

    double prod_comb = (sum_comb_x*sum_comb_y)/n/(n-1.0)*2.0; // expected sum_comb,
                                        // see Eq.(2) in (Hubert, Arabie, 1985)
    double mean_comb = (sum_comb_x+sum_comb_y)*0.5;
    double e_fm = prod_comb/sqrt(sum_comb_x*sum_comb_y); // expected FM

    CComparePartitionsPairsResult res;
    res.ar  = (sum_comb-prod_comb)/(mean_comb-prod_comb);
    res.r   = 1.0 + (2.0*sum_comb - (sum_comb_x+sum_comb_y))/n/(n-1.0)*2.0;
    res.fm  = sum_comb/sqrt(sum_comb_x*sum_comb_y);
    res.afm = (res.fm - e_fm)/(1.0 - e_fm); // Eq.(4) in (Hubert, Arabie, 1985)

    return res;
}




/*! Computes the mutual information-based scores
 *  for a given confusion matrix.
 *
 *  References
 *  ==========
 *
 *  Vinh, N.X., Epps, J., Bailey, J.,
 *  Information theoretic measures for clusterings comparison:
 *  Variants, properties, normalization and correction for chance,
 *  Journal of Machine Learning Research 11, 2010, pp. 2837-2854.
 *
 *  @param C a c_contiguous confusion matrix of size xc*yc
 *  @param xc number of rows in C
 *  @param yc number of columns in C
 *
 *  @return the computed scores
 */
template<class T>
CComparePartitionsInfoResult Ccompare_partitions_info(const T* C,
    Py_ssize_t xc, Py_ssize_t yc)
{
    double n = 0.0; // total sum (length of the underlying x and y = number of points)
    for (Py_ssize_t ij=0; ij<xc*yc; ++ij)
        n += C[ij];

    std::vector<double> sum_x(xc);
    std::vector<double> sum_y(yc);

    double h_x = 0.0, h_y = 0.0, h_x_cond_y = 0.0, h_x_y = 0.0;

    for (Py_ssize_t i=0; i<xc; ++i) {
        double t = 0.0;
        for (Py_ssize_t j=0; j<yc; ++j) {
            if (C[i*yc+j] > 0) h_x_y += C[i*yc+j]*std::log((double)C[i*yc+j]/(double)n);
            t += C[i*yc+j];
        }
        sum_x[i] = t;
        if (t > 0) h_y += t*std::log((double)t/(double)n);
    }

    for (Py_ssize_t j=0; j<yc; ++j) {
        double t = 0.0;
        for (Py_ssize_t i=0; i<xc; ++i) {
            if (C[i*yc+j] > 0) h_x_cond_y += C[i*yc+j]*std::log((double)C[i*yc+j]/sum_x[i]);
            t += C[i*yc+j];
        }
        sum_y[j] = t;
        if (t > 0) h_x += t*std::log((double)t/(double)n);
    }

    h_x = -h_x/(double)n;
    h_y = -h_y/(double)n;
    h_x_cond_y = -h_x_cond_y/(double)n;
    h_x_y = -h_x_y/(double)n;

    double e_mi = 0.0;
    for (Py_ssize_t i=0; i<xc; ++i) {
        double fac0 = lgamma(sum_x[i]+1.0)+lgamma(n-sum_x[i]+1.0)-lgamma(n+1.0);
        for (Py_ssize_t j=0; j<yc; ++j) {
            double fac1 = std::log((double)n/sum_x[i]/sum_y[j]);
            double fac2 = fac0+lgamma(sum_y[j]+1.0)+lgamma(n-sum_y[j]+1.0);

            for (Py_ssize_t nij=std::max(1.0, sum_x[i]+sum_y[j]-n);
                        nij<=std::min(sum_x[i],sum_y[j]); nij++) {
                double fac3 = fac2;
                fac3 -= lgamma(nij+1.0);
                fac3 -= lgamma(sum_x[i]-nij+1.0);
                fac3 -= lgamma(sum_y[j]-nij+1.0);
                fac3 -= lgamma(n-sum_x[i]-sum_y[j]+nij+1.0);
                e_mi += nij*(fac1+std::log((double)nij))*exp(fac3);
            }
        }
    }
    e_mi = e_mi/(double)n;

    CComparePartitionsInfoResult res;
    res.mi = h_x-h_x_cond_y;
    res.nmi = res.mi/(0.5*(h_x+h_y)); // NMI_sum in (Vinh et al., 2010)
    res.ami = (res.mi - e_mi)/(0.5*(h_x+h_y) - e_mi); // AMI_sum in (Vinh et al., 2010)

    return res;
}





/*! Computes the normalised pivoted accuracy
 *
 *  Normalised pivoted accuracy is
 *  (Accuracy(C[:,sigma])-1.0/max(xc,yc))/(1.0-1.0/max(xc,yc)),
 *  where C[:,sigma] is a version of the input confusion matrix
 *  with columns permuted based on the solution to the
 *  maximal linear sum assignment problem.
 *
 *  For non-square matrices, missing rows/columns are assumed
 *  to be filled with 0s.
 *
 *  Accuracy(C[:,sigma]) is sometimes referred to as
 *  set-matching classification rate or pivoted accuracy.
 *
 *
 *  References
 *  ==========
 *
 *  Steinley, D., Properties of the Hubert-Arabie adjusted Rand index,
 *  Psychological Methods 9(3), 2004, pp. 386-396,
 *  DOI:10.1037/1082-989X.9.3.386.
 *
 *  Meila, M., Heckerman, D., An experimental comparison of model-based clustering
 *  methods, Machine Learning 42, 2001, pp. 9-29, DOI:10.1023/A:1007648401407.
 *
 *  Gagolewski, M., Normalised clustering accuracy: An asymmetric external
 *  cluster validity measure, Journal of Classification 42, 2025, 2-30.
 *  DOI:10.1007/s00357-024-09482-2.
 *
 *
 *  @param C a c_contiguous confusion matrix of size xc*yc
 *  @param xc number of rows in C
 *  @param yc number of columns in C
 *
 *  @return the computed score
 */
template<class T>
double Ccompare_partitions_npa(const T* C, Py_ssize_t xc, Py_ssize_t yc)
{
    double n = 0.0; // total sum (length of the underlying x and y = number of points)
    for (Py_ssize_t ij=0; ij<xc*yc; ++ij) {
        if (C[ij] > 0) {
            n += C[ij];
        }
    }

    // if C is not a square matrix, treat the missing columns
    // as if they were filled with 0s
    Py_ssize_t xyc = std::max(xc, yc);
    std::vector<double> S(xyc*xyc, 0.0);
    for (Py_ssize_t i=0; i<xc; ++i) {
        for (Py_ssize_t j=0; j<yc; ++j) {
            if (C[i*yc+j] > 0) {
                S[i*xyc+j] = (double)C[i*yc+j];
            }
        }
    }

    std::vector<Py_ssize_t> output_col4row(xyc);

    Py_ssize_t retval = linear_sum_assignment(S.data(), xyc, xyc,
        output_col4row.data(), false); // minimise=false
    GENIECLUST_ASSERT(retval == 0);

    // sum of pivots:
    double t = 0.0;
    for (Py_ssize_t i=0; i<xyc; ++i)
        t += S[xyc*i+output_col4row[i]];

    double a = (double)t/(double)n;
    return (a*xyc-1.0)/(xyc-1.0);

}



/*! Computes the normalised clustering accuracy (NCA) score
 *
 *  NCA is not symmetric - we assume that rows in C determine the true
 *  (reference) partition.
 *
 *  For non-square confusion matrices, missing rows/columns
 *  are assumed to be filled with 0s and that 0/0 is 0,
 *  but the original row count is used for normalisation.
 *
 *  References
 *  ==========
 *
 *  Gagolewski, M., Normalised clustering accuracy: An asymmetric external
 *  cluster validity measure, Journal of Classification 42, 2025, 2-30.
 *  DOI:10.1007/s00357-024-09482-2.
 *
 *
 *  @param C a c_contiguous confusion matrix of size xc*yc
 *  @param xc number of rows in C
 *  @param yc number of columns in C
 *
 *  @return the computed score
 */
template<class T>
double Ccompare_partitions_nca(const T* C, Py_ssize_t xc, Py_ssize_t yc)
{
    std::vector<double> sum_x(xc, 0.0);
    for (Py_ssize_t i=0; i<xc; ++i) {
        for (Py_ssize_t j=0; j<yc; ++j) {
            if (C[i*yc+j] > 0) {
                sum_x[i] += C[i*yc+j];
            }
        }
    }

    // if xc>yc, treat C as if its missing columns were filled with 0s
    Py_ssize_t yc2 = std::max(xc, yc);

    // if xc<yc, only xc items are matched;
    // thus, overall, the behaviour is like filling missed rows/columns with 0s
    // and assuming 0/0 == 0, while still using k=nrow(C)

    std::vector<double> S(xc*yc2, 0.0);
    for (Py_ssize_t i=0; i<xc; ++i) {
        for (Py_ssize_t j=0; j<yc; ++j) {
            if (C[i*yc+j] > 0) {
                S[i*yc2+j] = (double)C[i*yc+j]/(double)sum_x[i];
            }
        }
    }

    std::vector<Py_ssize_t> output_col4row2(xc);
    Py_ssize_t retval = linear_sum_assignment(S.data(), xc, yc2,
        output_col4row2.data(), false); // minimise=false
    GENIECLUST_ASSERT(retval == 0);

    // sum of pivots
    double t = 0.0;
    for (Py_ssize_t i=0; i<xc; ++i)
        t += S[yc2*i+output_col4row2[i]];

    return (t-1.0)/(xc-1.0);
}


/*! Computes the pair sets index (PSI) and its simplified version,
 *  but without clipping negative values to 0.
 *
 *
 *  SPSI (simplified PSI) assumes E=1 in the definition of the index
 *  in (Rezaei, Franti 2016), i.e., uses Eq. (20) instead of Eq. (18) therein.
 *
 *  For non-square confusion matrices, missing rows/columns
 *  are assumed to be filled with 0s.
 *
 *
 *  References
 *  ==========
 *
 *  Rezaei, M., Franti, P., Set matching measures for external cluster validity,
 *  IEEE Transactions on Knowledge and Data Mining 28(8), 2016, pp. 2173-2186,
 *  DOI: 10.1109/TKDE.2016.2551240
 *
 *
 *  @param C a c_contiguous confusion matrix of size xc*yc
 *  @param xc number of rows in C
 *  @param yc number of columns in C
 *
 *  @return the computed scores
 */
template<class T>
CCompareSetMatchingResult Ccompare_partitions_psi(
    const T* C, Py_ssize_t xc, Py_ssize_t yc
) {

    double n = 0.0; // total sum (length of the underlying x and y = number of points)
    for (Py_ssize_t ij=0; ij<xc*yc; ++ij) {
        if (C[ij] > 0) {
            n += C[ij];
        }
    }

    // If C is not a square matrix, treat the missing columns or rows
    // as if they were filled with 0s.
    Py_ssize_t xyc = std::max(xc, yc);

    std::vector<double> sum_x(xyc, 0.0);
    std::vector<double> sum_y(xyc, 0.0);
    for (Py_ssize_t i=0; i<xc; ++i) {
        for (Py_ssize_t j=0; j<yc; ++j) {
            if (C[i*yc+j] > 0) {
                sum_x[i] += C[i*yc+j];
                sum_y[j] += C[i*yc+j];
            }
        }
    }

    std::vector<double> S(xyc*xyc, 0.0);
    for (Py_ssize_t i=0; i<xc; ++i) {
        for (Py_ssize_t j=0; j<yc; ++j) {
            if (C[i*yc+j] > 0) {
                S[i*xyc+j] = (double)C[i*yc+j]/(double)std::max(sum_x[i], sum_y[j]);
            }
        }
    }

    std::vector<Py_ssize_t> output_col4row2(xyc);
    Py_ssize_t retval = linear_sum_assignment(S.data(), xyc, xyc,
        output_col4row2.data(), false); // minimise=false
    GENIECLUST_ASSERT(retval == 0);

    // // sum of pivots:
    //     double s = 0.0;
    //     for (Py_ssize_t i=0; i<xyc; ++i)
    //         s += S[xyc*i+output_col4row2[i]];
    // from the smallest to the greatest is more numerically well-behaving:
    std::vector<double> pivots(xyc, 0.0);
    for (Py_ssize_t i=0; i<xyc; ++i)
        pivots[i] = S[xyc*i+output_col4row2[i]];
    std::sort(pivots.begin(), pivots.end());
    double s = 0.0;
    for (Py_ssize_t i=0; i<xyc; ++i)
        s += pivots[i];

    double es;
    std::sort(sum_x.begin(), sum_x.end());
    std::sort(sum_y.begin(), sum_y.end());
    es = 0.0;
    for (Py_ssize_t i=0; i<xyc; ++i) {
        //es += sum_y[xyc-i-1]*sum_x[xyc-i-1]/(double)std::max(sum_x[xyc-i-1], sum_y[xyc-i-1]);
        if (sum_y[i] > sum_x[i])
            es += sum_x[i];
        else
            es += sum_y[i];
    }
    es /= (double)n;

    CCompareSetMatchingResult res;

    // PSI uses max(0, PSI_unclipped)
    res.psi_unclipped = (s-es)/(xyc-es);
    res.spsi_unclipped = (s-1.0)/(xyc-1.0);

    return res;
}


#endif
