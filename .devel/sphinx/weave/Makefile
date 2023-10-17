# Copyright (c) 2013-2023, Marek Gagolewski <https://www.gagolewski.com>

FILES_RMD = \
	basics.Rmd              \
	sklearn_toy_example.Rmd \
	r.Rmd


FILES_RSTW = \
	benchmarks_ar.rstw       \
	benchmarks_details.rstw  \
	benchmarks_approx.rstw   \
	noise.rstw               \
	timings.rstw

#	string.rstw              \
#	sparse.rstw              \


RMD_MD_OUTPUTS=$(patsubst %.Rmd,%.md,$(FILES_RMD))
#RMD_RST_OUTPUTS=$(patsubst %.Rmd,%.rst,$(FILES_RMD))

RSTW_RST_OUTPUTS=$(patsubst %.rstw,%.rst,$(FILES_RSTW))

%.md: %.Rmd
	./Rmd2md.sh "$<"

#%.rst: %.md
#	pandoc -f markdown+grid_tables --wrap=none "$<" -o "$@"

%.rst: %.rstw
	./pweave_custom.py "$<" "$@"


all : rmd rstw

rmd : $(RMD_MD_OUTPUTS)

rstw : $(RSTW_RST_OUTPUTS)

clean:
	rm -f $(RSTW_RST_OUTPUTS) $(RMD_MD_OUTPUTS)
