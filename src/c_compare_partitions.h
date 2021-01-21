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
 *  Copyleft (C) 2018-2021, Marek Gagolewski <https://www.gagolewski.com>
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
void Cminmax(const T* x, ssize_t n, T* xmin, T* xmax)
{
    *xmin = x[0];
    *xmax = x[0];

    for (ssize_t i=1; i<n; ++i) {
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





/*! Applies partial pivoting to a given confusion matrix - permutes the columns
 *  so as to have the largest elements in each row on the main diagonal.
 *
 *  This comes in handy whenever C actually summarises the results generated
 *  by clustering algorithms, where actual label values do not matter
 *  (e.g., (1, 2, 0) can be remapped to (0, 2, 1) with no change in meaning.
 *
 *
 * @param C [in/out] a c_contiguous confusion matrix of size xc*yc
 * @param xc number of rows in C
 * @param yc number of columns in C
 *
 * Note that C is modified in-place (overwritten).
 */
template<class T>
void Capply_pivoting(T* C, ssize_t xc, ssize_t yc)
{
    for (ssize_t i=0; i<std::min(xc-1, yc-1); ++i) {
        ssize_t w = i;
        for (ssize_t j=i+1; j<yc; ++j) {
            // find w = argmax C[i,w], w=i,i+1,...yc-1
            if (C[i*yc+w] < C[i*yc+j]) w = j;
        }
        for (ssize_t j=0; j<xc; ++j) {
            // swap columns i and w
            std::swap(C[j*yc+i], C[j*yc+w]);
        }
    }
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
template<class T>
void Ccontingency_table(T* C, ssize_t xc, ssize_t yc,
        T xmin, T ymin,
        const T* x, const T* y, ssize_t n)
{
    for (ssize_t j=0; j<xc*yc; ++j)
        C[j] = 0;

    for (ssize_t i=0; i<n; ++i) {
        GENIECLUST_ASSERT(   0 <= (x[i]-xmin)*yc +(y[i]-ymin));
        GENIECLUST_ASSERT(xc*yc > (x[i]-xmin)*yc +(y[i]-ymin));
        C[(x[i]-xmin)*yc +(y[i]-ymin)]++;
    }
}


/*! Computes the adjusted and nonadjusted Rand- and FM scores
 *  based on a given confusion matrix.
 *
 *  References
 *  ==========
 *
 *  Hubert L., Arabie P., Comparing Partitions,
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
    ssize_t xc, ssize_t yc)
{
    double n = 0.0; // total sum (length of the underlying x and y = number of points)
    for (ssize_t ij=0; ij<xc*yc; ++ij)
        n += C[ij];

    double sum_comb_x = 0.0, sum_comb_y = 0.0, sum_comb = 0.0;

    for (ssize_t i=0; i<xc; ++i) {
        double t = 0.0;
        for (ssize_t j=0; j<yc; ++j) {
            t += C[i*yc+j];
            sum_comb += Ccomb2(C[i*yc+j]);
        }
        sum_comb_x += Ccomb2(t);
    }

    for (ssize_t j=0; j<yc; ++j) {
        double t = 0.0;
        for (ssize_t i=0; i<xc; ++i) {
            t += C[i*yc+j];
        }
        sum_comb_y += Ccomb2(t);
    }

    double prod_comb = (sum_comb_x*sum_comb_y)/n/(n-1.0)*2.0; // expected sum_comb,
                                        // see Eq.(2) in (Hubert, Arabie, 1985)
    double mean_comb = (sum_comb_x+sum_comb_y)*0.5;
    double e_fm = prod_comb/sqrt(sum_comb_x*sum_comb_y); // expected FM (variant)

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
 *  Vinh N.X., Epps J., Bailey J.,
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
    ssize_t xc, ssize_t yc)
{
    double n = 0.0; // total sum (length of the underlying x and y = number of points)
    for (ssize_t ij=0; ij<xc*yc; ++ij)
        n += C[ij];

    std::vector<double> sum_x(xc);
    std::vector<double> sum_y(yc);

    double h_x = 0.0, h_y = 0.0, h_x_cond_y = 0.0, h_x_y = 0.0;

    for (ssize_t i=0; i<xc; ++i) {
        double t = 0.0;
        for (ssize_t j=0; j<yc; ++j) {
            if (C[i*yc+j] > 0) h_x_y += C[i*yc+j]*std::log((double)C[i*yc+j]/(double)n);
            t += C[i*yc+j];
        }
        sum_x[i] = t;
        if (t > 0) h_y += t*std::log((double)t/(double)n);
    }

    for (ssize_t j=0; j<yc; ++j) {
        double t = 0.0;
        for (ssize_t i=0; i<xc; ++i) {
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
    for (ssize_t i=0; i<xc; ++i) {
        double fac0 = lgamma(sum_x[i]+1.0)+lgamma(n-sum_x[i]+1.0)-lgamma(n+1.0);
        for (ssize_t j=0; j<yc; ++j) {
            double fac1 = std::log((double)n/sum_x[i]/sum_y[j]);
            double fac2 = fac0+lgamma(sum_y[j]+1.0)+lgamma(n-sum_y[j]+1.0);

            for (ssize_t nij=std::max(1.0, sum_x[i]+sum_y[j]-n);
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





/*! Computes the normalised accuracy score between two partitions
 *
 *  Normalised accuracy is (Accuracy(C[sigma])-1.0/yc)/(1.0-1.0/yc),
 *  where C[sigma] is a version of the input confusion matrix
 *  with columns permuted based on the solution to the
 *  maximal linear sum assignment problem.
 *
 *  Accuracy(C[sigma]) is sometimes referred to as purity,
 *  e.g. in (Rendon et al. 2011).
 *
 *
 *  References
 *  ==========
 *
 *  Rendon E., Abundez I., Arizmendi A., Quiroz E.M.,
 *  Internal versus external cluster validation indexes,
 *  International Journal of Computers and Communications 5(1), 2011, pp. 27-34.

 *
 *  @param C a c_contiguous confusion matrix of size xc*yc
 *  @param xc number of rows in C, xc <= yc
 *  @param yc number of columns in C
 *
 *  @return the computed score
 */
template<class T>
double Ccompare_partitions_nacc(const T* C, ssize_t xc, ssize_t yc)
{
    GENIECLUST_ASSERT(xc <= yc);

    double n = 0.0; // total sum (length of the underlying x and y = number of points)
    for (ssize_t ij=0; ij<xc*yc; ++ij)
        n += C[ij];

    std::vector<ssize_t> output_col4row(xc);

    ssize_t retval = linear_sum_assignment(C, xc, yc, output_col4row.data(), false); // minimise=false
    GENIECLUST_ASSERT(retval == 0);

    double t = 0.0;
    for (ssize_t i=0; i<xc; ++i)
        t += C[yc*i+output_col4row[i]];

    double pur = (double)t/(double)n;
    return (pur-1.0/yc)/(1.0-1.0/yc);

}





/*! Computes the PSI (pair sets index) score
 *
 *  References
 *  ==========
 *
 *  Rezaei M., Franti P., Set matching measures for external cluster validity,
 *  IEEE Transactions on Knowledge and Data Mining 28(8), 2016, pp. 2173-2186,
 *  doi:10.1109/TKDE.2016.2551240
 *
 *  @param C a c_contiguous confusion matrix of size xc*yc
 *  @param xc number of rows in C, xc <= yc
 *  @param yc number of columns in C
 *
 *  @return the computed score
 */
template<class T>
double Ccompare_partitions_psi(const T* C, ssize_t xc, ssize_t yc)
{
    GENIECLUST_ASSERT(xc <= yc);

    double n = 0.0; // total sum (length of the underlying x and y = number of points)
    for (ssize_t ij=0; ij<xc*yc; ++ij)
        n += C[ij];

    std::vector<double> sum_x(xc);
    std::vector<double> sum_y(yc);
    for (ssize_t i=0; i<xc; ++i) {
        for (ssize_t j=0; j<yc; ++j) {
            sum_x[i] += C[i*yc+j];
            sum_y[j] += C[i*yc+j];
        }
    }

    std::vector<double> S(xc*yc);
    for (ssize_t i=0; i<xc; ++i) {
        for (ssize_t j=0; j<yc; ++j) {
            S[i*yc+j] = (double)C[i*yc+j]/(double)std::max(sum_x[i], sum_y[j]);
        }
    }
    std::vector<ssize_t> output_col4row2(xc);
    ssize_t retval = linear_sum_assignment(S.data(), xc, yc, output_col4row2.data(), false); // minimise=false
    GENIECLUST_ASSERT(retval == 0);

    double s = 0.0;
    for (ssize_t i=0; i<xc; ++i)
        s += S[yc*i+output_col4row2[i]];

    std::sort(sum_x.begin(), sum_x.end());
    std::sort(sum_y.begin(), sum_y.end());
    double es = 0.0;
    for (ssize_t i=0; i<xc; ++i)
        es += sum_y[yc-i-1]*sum_x[xc-i-1]/(double)std::max(sum_x[xc-i-1], sum_y[yc-i-1]);
    es /= (double)n;

    double psi  = (s-es)/(yc-es);
    if (psi<0.0) psi = 0.0;

    return psi;
}


#endif
