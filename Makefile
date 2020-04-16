# (C) 2020 Marek Gagolewski, https://www.gagolewski.com

#VPATH="/home/gagolews/Python/genieclust"


.PHONY: all user

all: please_specify_build_target_manually

user:
	CPPFLAGS="-fopenmp -march=native -mtune=native" \
	LDFLAGS="-fopenmp" python3 setup.py install --user
