ReportName = Final_Project_Proposal_CKK
NoteBooks = 
References = references.bib

$(ReportName).pdf:$(ReportName).md $(NoteBooks:.ipynb=.pdf) $(References)
	pandoc -s -f markdown-implicit_figures $(ReportName).md -o $(ReportName).pdf --bibliography $(References) --citeproc --verbose --csl=ieee.csl

clean:
	rm $(ReportName).pdf $(NoteBooks:.ipynb=.pdf)

# https://nbconvert.readthedocs.io/en/latest/usage.html
#Rule for rendering iPython notebooks to pdf
$(NoteBooks:.ipynb=.pdf): $(NoteBooks)
	jupyter nbconvert --to PDF $(NoteBooks)
