# Copyright (c) 2013-2025, Marek Gagolewski <https://www.gagolewski.com>

FILES_RMD = \
	basics.Rmd              \
	sklearn_toy_example.Rmd \
	noise.Rmd               \
	r.Rmd                   \
	benchmarks_ar.Rmd       \
	timings.Rmd

# 	benchmarks_details.Rmd  \
#	benchmarks_approx.Rmd   \
#	sparse.Rmd              \
#	string.Rmd              \


RMD_MD_OUTPUTS=$(patsubst %.Rmd,%.md,$(FILES_RMD))

%.md: %.Rmd
	./Rmd2md.sh "$<"

all : rmd

rmd : $(RMD_MD_OUTPUTS)

clean:
	rm -f $(RMD_MD_OUTPUTS)
