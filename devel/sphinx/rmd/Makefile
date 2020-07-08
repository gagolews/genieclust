FILES_RMD = \
	r.Rmd


FILES_RSTW = \
	basics.rstw              \
	benchmarks_ar.rstw       \
	sklearn_toy_example.rstw \
	benchmarks_details.rstw  \
	noise.rstw


RMD_MD_OUTPUTS=$(patsubst %.Rmd,%.md,$(FILES_RMD))
RMD_RST_OUTPUTS=$(patsubst %.Rmd,%.rst,$(FILES_RMD))

RSTW_RST_OUTPUTS=$(patsubst %.rstw,%.rst,$(FILES_RSTW))

%.md: %.Rmd
	Rscript -e 'knitr::knit("$<", "$@")'

%.rst: %.md
	pandoc -f markdown+grid_tables --wrap=none "$<" -o "$@"

%.rst: %.rstw
	pweave "$<" -o "$@"


all : rmd rstw

rmd : $(RMD_MD_OUTPUTS) $(RMD_RST_OUTPUTS)

rstw : $(RSTW_RST_OUTPUTS)

