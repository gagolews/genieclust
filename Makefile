# (C) 2020 Marek Gagolewski, https://www.gagolewski.com

#VPATH="/home/gagolews/Python/genieclust"


.PHONY: python pytest r check testthat

all: please_specify_build_target_manually

#CPPFLAGS="-fopenmp -march=native -mtune=native"
#LDFLAGS="-fopenmp"

python:
	python3 setup.py install --user

pytest: python
	pytest

r:
	Rscript -e 'Rcpp::compileAttributes()'
	R CMD INSTALL .
	# AVOID ADDING THE -O0 flag!!!
	Rscript -e 'roxygen2::roxygenise(roclets=c("rd", "collate", "namespace", "vignette"), load_code=roxygen2::load_installed)'
	R CMD INSTALL .

check: r
	Rscript -e 'devtools::check()'

testthat: r
	Rscript -e 'options(width=120); source("devel/testthat.R")'
