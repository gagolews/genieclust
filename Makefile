# Copyleft (C) 2020-2021, Marek Gagolewski <https://www.gagolewski.com>


.PHONY: python py-test py-check r r-check r r-build sphinx clean

PKGNAME="genieclust"

all: r python

#CPPFLAGS="-fopenmp -march=native -mtune=native"
#LDFLAGS="-fopenmp"

python:
	python3 setup.py install --user

py-test: python
	pytest
	cd devel/sphinx && make doctest && cd ../../

weave:
	cd devel/sphinx/weave && make && cd ../../../

rd2rst:
	# https://github.com/gagolews/Rd2rst
	# TODO: if need be, you can also use MyST in the future
	cd devel/sphinx && Rscript -e "Rd2rst::Rd2rst('${PKGNAME}')" && cd ../../

news:
	cd devel/sphinx && pandoc ../../NEWS -f markdown -t rst -o news.rst

sphinx: python r weave rd2rst news
	rm -rf devel/sphinx/_build/
	cd devel/sphinx && make html && cd ../../
	rm -rf docs/
	mkdir docs/
	cp -rf devel/sphinx/_build/html/* docs/
	cp devel/CNAME.tpl docs/CNAME
	touch docs/.nojekyll
	touch .nojekyll

py-check: python
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics \
		--exclude=devel,build,docs,.git,R,dist,genieclust.egg-info,man,tutorials
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 \
		--statistics  \
		--exclude=devel,build,docs,.git,R,dist,genieclust.egg-info,man,tutorials \
		--ignore=E121,E123,E126,E226,E24,E704,W503,W504,E221,E303,E265

r-autoconf:
	Rscript -e 'Rcpp::compileAttributes()'
	Rscript -e 'roxygen2::roxygenise(roclets=c("rd", "collate", "namespace", "vignette"), load_code=roxygen2::load_installed)'

r: r-autoconf
	R CMD INSTALL . --html

r-test: r
	Rscript -e 'source("devel/tinytest.R")'

r-build:
	cd .. && R CMD build ${PKGNAME}

r-check: r-build
	cd .. && _R_CHECK_FORCE_SUGGESTS_=0 R CMD check `ls -t ${PKGNAME}*.tar.gz | head -1` --no-manual --as-cran

clean:
	python3 setup.py clean
	rm -rf genieclust/__pycache__/
	rm -rf genieclust.egg-info/
	rm -rf dist/
	rm -f genieclust/*.cpp
	find src -name '*.o' -exec rm {} \;
	find src -name '*.so' -exec rm {} \;

purge: clean
	rm -f man/*.Rd
