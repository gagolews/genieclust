# Copyleft (C) 2020-2023, Marek Gagolewski <https://www.gagolewski.com>

.PHONY: python py-test py-check r r-check r r-build sphinx docs clean

.NOTPARALLEL: python py-test py-check r r-check r r-build clean purge sphinx docs

PKGNAME="genieclust"

all: r python

################################################################################

python:
	# python3 setup.py install --user  # deprecated
	# python3 -m build  # requires the 'build' package, creates an isolated envir
	python3 -m pip install .

py-test: python
	pytest
	cd .devel/sphinx && make doctest && cd ../../

py-check: python
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics \
		--exclude=.devel,build,docs,.git,R,dist,genieclust.egg-info,man,tutorials
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 \
		--statistics  \
		--exclude=.devel,build,docs,.git,R,dist,genieclust.egg-info,man,tutorials \
		--ignore=E121,E123,E126,E226,E24,E704,W503,W504,E221,E303,E265

################################################################################

stop-on-utf8:
	# Stop if some files are not in ASCII:
	[ -z "`file -i DESCRIPTION configure configure.win \
	        NAMESPACE cleanup R/* src/* man/* inst/* tools/* | \
	    grep 'text/' | grep -v 'us-ascii' | tee /dev/stderr`" ]

r-autoconf:
	Rscript -e 'Rcpp::compileAttributes()'
	Rscript -e "\
	    source('.devel/roxygen2-patch.R');\
	    roxygenise(\
	        roclets=c('rd', 'collate', 'namespace', 'vignette'),\
	    )"
# 	        load_code=roxygen2::load_installed\

r: r-autoconf
	R CMD INSTALL . --html

r-test: r
	Rscript -e 'source(".devel/tinytest.R")'

r-build: r-autoconf
	cd .. && R CMD build ${PKGNAME}

r-check: stop-on-utf8 r-build
	cd .. && R_DEFAULT_INTERNET_TIMEOUT=240 \
	    _R_CHECK_CRAN_INCOMING_REMOTE_=FALSE \
	    _R_CHECK_FORCE_SUGGESTS_=0 R CMD check `ls -t ${PKGNAME}*.tar.gz | head -1` --no-manual --as-cran

################################################################################

rd2myst:
	cd .devel/sphinx && Rscript -e "Rd2rst::Rd2myst('${PKGNAME}')"

weave-examples:
	cd .devel/sphinx/rapi && Rscript -e "Rd2rst::weave_examples('${PKGNAME}', '.')"
	.devel/sphinx/fix-code-blocks.sh .devel/sphinx/rapi

weave:
	cd .devel/sphinx/weave && make && cd ../../../

news:
	cd .devel/sphinx && cp ../../NEWS news.md

html: python r news weave rd2myst weave-examples
	rm -rf .devel/sphinx/_build/
	cd .devel/sphinx && make html
	rm -rf .devel/sphinx/_build/html/_sources
	@echo "*** Browse the generated documentation at"\
	    "file://`pwd`/.devel/sphinx/_build/html/index.html"

docs: html
	@echo "*** Making 'docs' is only recommended when publishing the"\
	    "official release, because it updates the package homepage."
	@echo "*** Therefore, we check if the package version is like 1.2.3"\
	    "and not 1.2.2.9007."
	#Rscript --vanilla -e "stopifnot(length(unclass(packageVersion('${PKGNAME}'))[[1]]) < 4)"
	rm -rf docs/
	mkdir docs/
	cp -rf .devel/sphinx/_build/html/* docs/
	cp .devel/CNAME.tpl docs/CNAME
	touch docs/.nojekyll
	touch .nojekyll

################################################################################

clean:
	python3 setup.py clean
	rm -rf genieclust/__pycache__/
	rm -rf genieclust.egg-info/
	rm -rf ../genieclust.Rcheck/
	rm -rf dist/
	rm -rf build/
	rm -rf .devel/pytest/__pycache__/
	rm -f genieclust/*.cpp
	find src -name '*.o' -exec rm {} \;
	find src -name '*.so' -exec rm {} \;
	rm -rf .devel/sphinx/_build/
	rm -rf .devel/sphinx/rapi/
	rm -rf revdep/

purge: clean
	#rm -f man/*.Rd
