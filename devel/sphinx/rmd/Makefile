FILES_RMD = \
	sklearn_toy_example.Rmd \
	r.Rmd


RMD_MD_OUTPUTS=$(patsubst %.Rmd,%.md,$(FILES_RMD))
RMD_RST_OUTPUTS=$(patsubst %.Rmd,%.rst,$(FILES_RMD))

%.md: %.Rmd
	Rscript -e 'knitr::knit("$<", "$@")'

%.rst: %.md
	pandoc "$<" -o "$@"

rmd : $(RMD_MD_OUTPUTS) $(RMD_RST_OUTPUTS)


