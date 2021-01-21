/*  class CIntDict
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


#ifndef __c_int_dict_h
#define __c_int_dict_h

#include "c_common.h"
#include <algorithm>
#include <vector>
#include <iterator>



/*! ordered_map (dictionary) for keys in {0,1,...,n-1} (small ints).
 * Elements are stored in the natural 0 <= 1 <= ... <= n-1 order.
 *
 * Most supported operations (except for inserting a new key
 * in the "middle") are as fast as you can get.
 * Yet, everything comes at a price: here it's the O(n) memory
 * requirement, even if data are sparse.
 *
 * Use case: GiniDisjointSets in the `genieclust` Python package.
 */
template <class T>
class CIntDict {

protected:
    ssize_t n;                //!< total number of distinct keys possible
    ssize_t k;                //!< number of keys currently stored
    std::vector<T> tab;       //!< tab[i] is the element associated with key i

    std::vector<ssize_t> tab_next; //!< an array-based...
    std::vector<ssize_t> tab_prev; //!< ...doubly-linked list...
    ssize_t  tab_head; //!< ...for quickly accessing and iterating over...
    ssize_t  tab_tail; //!< ...this->tab data



public:
    /*! Constructs an empty container.
     *
     *  @param n number of elements, n>=0.
     */
    CIntDict(ssize_t n) :
        tab(n), tab_next(n, n), tab_prev(n, -1)
    {
        // if (n < 0) throw std::domain_error("n < 0");
        this->n = n;
        this->k = 0;
        this->tab_head = n;
        this->tab_tail = -1;
    }


    /*! Constructs a full-size container.
     *
     *  @param n number of elements, n>=0.
     *  @param val value to replicate at each position
     */
    CIntDict(ssize_t n, const T& val) :
        tab(n), tab_next(n), tab_prev(n)
    {
        // if (n < 0) throw std::domain_error("n < 0");
        this->n = n;
        this->k = n;
        for (ssize_t i=0; i<n; ++i) {
            this->tab_prev[i] = i-1;
            this->tab_next[i] = i+1;
            this->tab = val;
        }
        this->tab_head = 0;
        this->tab_tail = n-1;
    }


    /*! A nullary constructor allows Cython to allocate
     *  the instances on the stack. Do not use otherwise.
    */
    CIntDict() : CIntDict(0) { }


    /*! Returns the current number of elements in the container.
     *
     * Time complexity: O(1)
     */
    inline ssize_t size() const { return this->k; }


    /*! Returns the maximum number of elements that the container can hold.
     */
    inline ssize_t max_size() const { return this->n; }

    /*! Tests whether the container is empty, i.e., its size() is 0.
     */
    inline bool empty() const { return this->k == 0; }

    /*! Counts the number of elements with given key, i.e., returns 0 or 1
     * depending on whether an element with key i exists.
     *
     * Time complexity: O(1)
     *
     * @param i key in [0,n)
     */
    inline size_t count(ssize_t i) const {
        if (i < 0 || i >= n)
            throw std::out_of_range("CIntDict::count key out of range");
        return (tab_prev[i]>=0 || tab_next[i]<n || i==tab_head) ? 1 : 0;

    }

    /*! Clears the container's content.
     *
     * Time complexity: O(k), where k is the current number of elements
     * in the container.
     */
    void clear() {
        if (k == 0) return;

        ssize_t cur = tab_head;
        while (cur < n) {
            tab[cur] = T(); // force destructor call

            ssize_t last = cur;
            cur = tab_next[cur];

            tab_prev[last] = -1;
            tab_next[last] =  n;
        }
        k = 0;
        tab_head = n;
        tab_tail = -1;
    }


    /*! Returns a reference to the value mapped to key i.
     *  If key i does not exist, the function throws an out_of_range exception.
     *
     * Time complexity: O(1)
     *
     * @param i key in [0,n)
     */
    inline T& at(ssize_t i) {
        if (!count(i))
            throw std::out_of_range("CIntDict::at key does not exist");
        return tab[i];
    }

    inline const T& at(ssize_t i) const {
        if (!count(i))
            throw std::out_of_range("CIntDict::at key does not exist");
        return tab[i];
    }


    // inline const T& operator[] (ssize_t i) const {
    //     if (!count(i))       throw std::out_of_range("key does not exist");
    //     return tab[i];
    // }

    /*! Returns a reference to the value mapped to key i.
     *  If key i does not exist, a new element is inserted.
     *
     * Time complexity: accessing existing element is O(1),
     * inserting the smallest or the largest element is O(1), and
     * inserting in the "middle" is O(k), where k is the current number
     * of elements in the container.
     *
     * @param i key in [0,n)
     */
    T& operator[] (ssize_t i) {
        if (!count(i)) {
            // adding a new element

            if (k == 0) {
                // empty -> new head and tail
                tab_head = tab_tail = i;
            }
            else if (i < tab_head) {
                // new head
                tab_next[i] = tab_head;
                GENIECLUST_ASSERT(tab_prev[i] == -1);
                tab_prev[tab_head] = i;
                tab_head = i;
            }
            else if (i > tab_tail) {
                // new tail
                tab_next[tab_tail] = i;
                tab_prev[i] = tab_tail;
                GENIECLUST_ASSERT(tab_next[i] == n);
                tab_tail = i;
            }
            else {
                // insert in the "middle"
                // slow op
                // TODO skip list, etc. ??
                ssize_t elem_before_i = tab_head;
                while (tab_next[elem_before_i] < i)
                    elem_before_i = tab_next[elem_before_i];

                ssize_t elem_after_i = tab_next[elem_before_i];
                GENIECLUST_ASSERT(tab_prev[elem_after_i] == elem_before_i);
                tab_next[i] = elem_after_i;
                tab_prev[i] = elem_before_i;
                tab_next[elem_before_i] = i;
                tab_prev[elem_after_i]  = i;
            }

            k++; // we have a brand new elem in the storage
        }

        return tab[i];
    }


    /*! Removes a single element, provided it exists.
     *
     * Time complexity: O(1)
     *
     * @param i key in [0,n)
     * @return the number of elements removed (0 or 1)
     */
    ssize_t erase(ssize_t i) {
        if (!count(i))
            return 0;

        if (i == tab_head && i == tab_tail) {
            // that was the last (size-wise) element in the container
            tab_head = n;
            tab_tail = -1;
        }
        else if (i == tab_head) {
            // that was the least element
            tab_head = tab_next[tab_head];
            tab_prev[tab_head] = -1;
        }
        else if (i == tab_tail) {
            // that was the largest one
            tab_tail = tab_prev[tab_tail];
            tab_next[tab_tail] = n;
        }
        else {
            // elem in the "middle"
            ssize_t elem_after_i  = tab_next[i];
            ssize_t elem_before_i = tab_prev[i];
            tab_next[elem_before_i] = elem_after_i;
            tab_prev[elem_after_i]  = elem_before_i;
        }

        tab[i] = T(); // force destructor call
        tab_prev[i] = -1;
        tab_next[i] = n;
        k--;
        return 1; // one element has been removed
    }


    ssize_t get_key_min() const { return tab_head; }
    ssize_t get_key_max() const { return tab_tail; }
    ssize_t get_key_next(ssize_t i) const { return tab_next[i]; }
    ssize_t get_key_prev(ssize_t i) const { return tab_prev[i]; }

    ssize_t pop_key_min() {
        ssize_t ret = tab_head;
        erase(ret);
        return ret;
    }

    ssize_t pop_key_max() {
        ssize_t ret = tab_tail;
        erase(ret);
        return ret;
    }



    // ------- minimal iterator-based interface -----------------------------

    /*! If you want more than merely an input_iterator,
     * go ahead, implement it and make a pull request :)
     */
    class iterator : public std::iterator<std::input_iterator_tag, ssize_t> {
        private:
            const ssize_t* tab_next;
            ssize_t cur;
        public:
            iterator(ssize_t tab_head, ssize_t* tab_next) :
                tab_next(tab_next), cur(tab_head) { }
            iterator& operator++() { cur = tab_next[cur]; return *this; }
            iterator operator++(int) {
                iterator tmp(*this); operator++(); return tmp;
            }
            bool operator==(const iterator& rhs) const {
                return tab_next==rhs.tab_next && cur==rhs.cur;
            }
            bool operator!=(const iterator& rhs) const {
                return tab_next!=rhs.tab_next || cur!=rhs.cur;
            }
            ssize_t operator*() const { return cur; }
    };


    /*! Returns an iterator pointing to the element in the container
     * that has the least key
     */
    iterator begin() { return iterator(tab_head, tab_next.data()); }

    /*! Returns an iterator pointing to the past-the-end element
     */
    iterator end() { return iterator(n, tab_next.data()); }



    // TODO /go ahead, write it, make a pull request/
    //Returns an iterator pointing to the element in the container
    // that has the greatest key
    // reverse_iterator rbegin()

    // TODO /go ahead, write it, make a pull request/
    //Returns an iterator pointing to the past-the-beginning element
    // reverse_iterator rend()

    // TODO /go ahead, write it, make a pull request/
    // cbegin, cend, crbegin, crend()

    // TODO /go ahead, write it, make a pull request/
    // Returns an iterator to an element with given key or returns an iterator to end() if not exists
    //      iterator find ( const key_type& k );
    //const_iterator find ( const key_type& k ) const;

    // TODO /go ahead, write it, make a pull request/
    // Removes  a single element from the  container
    //iterator erase ( const_iterator position ); //by position (1)

};

#endif
