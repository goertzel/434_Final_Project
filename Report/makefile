FILE=report

all: $(FILE).pdf

.PHONY: clean


$(FILE).pdf: *.tex 
	pdflatex -shell-escape $(FILE).tex
	make clean

clean:
	rm -f *.aux *.log

cleanall:
	rm -f *.aux *.log *.pdf


