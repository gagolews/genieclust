# Copyright (c) 2013-2023, Marek Gagolewski <https://www.gagolewski.com>

FILES_RMD = \
	r.Rmd


FILES_RSTW = \
	sklearn_toy_example.rstw \
	basics.rstw              \
	benchmarks_ar.rstw       \
	benchmarks_details.rstw  \
	benchmarks_approx.rstw   \
	noise.rstw               \
	sparse.rstw              \
	string.rstw              \
	timings.rstw


RMD_MD_OUTPUTS=$(patsubst %.Rmd,%.md,$(FILES_RMD))
#RMD_RST_OUTPUTS=$(patsubst %.Rmd,%.rst,$(FILES_RMD))

RSTW_RST_OUTPUTS=$(patsubst %.rstw,%.rst,$(FILES_RSTW))

%.md: %.Rmd
	Rscript -e 'knitr::knit("$<", "$@")'

#%.rst: %.md
#	pandoc -f markdown+grid_tables --wrap=none "$<" -o "$@"

%.rst: %.rstw
	./pweave_custom.py "$<" "$@"


all : rmd rstw

rmd : $(RMD_MD_OUTPUTS)

rstw : $(RSTW_RST_OUTPUTS)

clean:
	rm -f $(RSTW_RST_OUTPUTS) $(RMD_MD_OUTPUTS)
