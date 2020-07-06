FILES_RMD = \
	sklearn_toy_example.Rmd \
	r.Rmd \
	benchmarks_ar.Rmd


RMD_MD_OUTPUTS=$(patsubst %.Rmd,%.md,$(FILES_RMD))
RMD_RST_OUTPUTS=$(patsubst %.Rmd,%.rst,$(FILES_RMD))

%.md: %.Rmd
	Rscript -e 'knitr::knit("$<", "$@")'

%.rst: %.md
	pandoc -f markdown+grid_tables --wrap=none "$<" -o "$@"

rmd : $(RMD_MD_OUTPUTS) $(RMD_RST_OUTPUTS)


