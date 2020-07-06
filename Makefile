# (C) 2020 Marek Gagolewski, https://www.gagolewski.com

#VPATH="/home/gagolews/Python/genieclust"


.PHONY: python py-test py-check r r-check r r-build sphinx

all: r python

#CPPFLAGS="-fopenmp -march=native -mtune=native"
#LDFLAGS="-fopenmp"

python:
	python3 setup.py install --user

py-test: python
	pytest
	cd devel/sphinx && make doctest && cd ../../

rmd:
	cd devel/sphinx/rmd && make && cd ../../../

sphinx: python r rmd
	rm -rf devel/sphinx/_build/
	Rscript -e "Rd2md::ReferenceManual()"
	Rscript -e "f <- readLines('Reference_Manual_genieclust.md');" \
		-e "f <- f[which(stringi::stri_detect_regex(f, '^#'))[2]:length(f)];" \
		-e "f <- stringi::stri_replace_first_regex(f, '^(#+) \\u0060?(.*?)\\u0060?\\u0024', '#\\u00241 \\u00242')" \
		-e "f <- stringi::stri_replace_first_regex(f, 'list\\\\(list\\\\(\"(.*?)\"\\\\), list\\\\(\"(.*?)\"\\\\)\\\\)', '\\u00241.\\u00242')" \
		-e "f <- c('# R Package *genieclust* Reference', '', f)" \
		-e "writeLines(f, 'r.md')"
	pandoc r.md -o devel/sphinx/r.rst
	#rstlisttable --in-place devel/sphinx/r.rst
	rm -f Reference_Manual_genieclust.md r.md
	cd devel/sphinx && make html && cd ../../
	rm -rf docs/
	mkdir docs/
	cp -rf devel/sphinx/_build/html/* docs/
	cp devel/CNAME.tpl docs/CNAME
	touch docs/.nojekyll

py-check: python
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=devel,build,docs,.git,R,dist,genieclust.egg-info,man,tutorials
	flake8 . --count --exit-zero --max-complexity=10 --ignore=E121,E123,E126,E226,E24,E704,W503,W504,E221,E303,E265 --max-line-length=127 --statistics  --exclude=devel,build,docs,.git,R,dist,genieclust.egg-info,man,tutorials

r:
	Rscript -e 'Rcpp::compileAttributes()'
	R CMD INSTALL .
	# AVOID ADDING THE -O0 flag!!!
	Rscript -e 'roxygen2::roxygenise(roclets=c("rd", "collate", "namespace", "vignette"), load_code=roxygen2::load_installed)'
	R CMD INSTALL .

r-check: r
	Rscript -e 'devtools::check()'

r-test: r
	Rscript -e 'options(width=120); source("devel/testthat.R")'


r-build:
	Rscript -e 'Rcpp::compileAttributes()'
	Rscript -e 'roxygen2::roxygenise(roclets=c("rd", "collate", "namespace", "vignette"))'
	R CMD INSTALL . --preclean
	R CMD build .
