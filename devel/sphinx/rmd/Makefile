FILES_RMD = \
	sklearn_toy_example.Rmd \
	r.Rmd \
	benchmarks_ar.Rmd \
	basics.Rmd


FILES_RSTW = \
	benchmarks_details.rstw


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

