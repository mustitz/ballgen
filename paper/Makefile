default: tables/numbers.tex
	latexmk -pdf trionda

# numbers.tex and the tables are generated from data/*.json; rebuild them when the
# data or the generator changes (never edit them by hand).
tables/numbers.tex: make_tables.py data/spectre.json
	python make_tables.py > /dev/null
