SPFIG = closed_balls.png joined_spheres.png sphere_chain.png sphere_star.png

default: tables/numbers.tex $(addprefix figures/,$(SPFIG))
	latexmk -pdf trionda

# numbers.tex and the tables are generated from data/*.json (never edit by hand).
tables/numbers.tex: make_tables.py data/spectre.json
	python make_tables.py > /dev/null

# Spectre figures are produced in ../spectre/figures; copy them next to the others.
figures/%.png: ../spectre/figures/%.png
	cp $< $@
