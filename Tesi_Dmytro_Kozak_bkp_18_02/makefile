# Nome del file principale senza estensione
finalname = main

# File sorgenti inclusi
included_latex_sources = \
	introduzione.tex \
	capitolo1.tex \
	capitolo2.tex \
	capitolo3.tex \
	capitolo4.tex \
	capitolo5.tex \
	conclusioni.tex \
	ringraziamenti.tex

# File bibliografia
bibfile = bibliografia.bib

# Comando per compilare (usa latexmk per gestire tutto automaticamente)
all: $(finalname).pdf

$(finalname).pdf: $(finalname).tex $(included_latex_sources) $(bibfile)
	latexmk -pdf -interaction=nonstopmode $(finalname).tex

clean:
	latexmk -C
	rm -f *.bbl *.run.xml *.synctex.gz
