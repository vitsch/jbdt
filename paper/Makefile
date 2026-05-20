# Build the Bayesian Analysis submission package
# Requires: pdflatex, bibtex, and imsart.cls (see README.md)

.PHONY: all main supplement cover clean distclean

all: main supplement cover

main: main.pdf

supplement: supplement.pdf

cover: cover_letter.pdf

main.pdf: main.tex references.bib
	pdflatex -interaction=nonstopmode main
	bibtex main
	pdflatex -interaction=nonstopmode main
	pdflatex -interaction=nonstopmode main

supplement.pdf: supplement.tex references.bib
	pdflatex -interaction=nonstopmode supplement
	bibtex supplement
	pdflatex -interaction=nonstopmode supplement
	pdflatex -interaction=nonstopmode supplement

cover_letter.pdf: cover_letter.tex
	pdflatex -interaction=nonstopmode cover_letter

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk

distclean: clean
	rm -f main.pdf supplement.pdf cover_letter.pdf
