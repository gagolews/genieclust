# (C) 2020 Marek Gagolewski, https://www.gagolewski.com

#VPATH="/home/gagolews/Python/genieclust"


.PHONY: python pytest r check testthat

all: please_specify_build_target_manually

python:
	CPPFLAGS="-fopenmp -march=native -mtune=native" \
	LDFLAGS="-fopenmp" python3 setup.py install --user

pytest: python
	pytest

r:
	Rscript -e 'Rcpp::compileAttributes()'
	Rscript -e 'devtools::document(roclets = c("rd", "collate", "namespace", "vignette"))'
	R CMD INSTALL .

check: r
	Rscript -e 'devtools::check()'

testthat: r
	Rscript -e 'options(width=120); source("devel/testthat.R")'


