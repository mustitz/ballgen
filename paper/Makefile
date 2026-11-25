SPFIG = closed_balls.png joined_spheres.png sphere_chain.png sphere_star.png
EXFIG = compare_views.png compare_ledger.png compare_spectra.png compare_exposure.png compare_hat_trionda.png compare_flow.png \
        hat_trick_ball.png hat_trick_hat.png

default: tables/numbers.tex $(addprefix figures/,$(SPFIG)) $(addprefix figures/,$(EXFIG))
	latexmk -pdf trionda

# numbers.tex and the tables are generated from data/*.json (never edit by hand).
tables/numbers.tex: make_tables.py data/spectre.json data/hat_trick.json
	python make_tables.py > /dev/null

# data/hat_trick.json comes from the repository's Hat Trick and comparison code (a few minutes); data/hat_frequencies.json
# is copied unchanged from the hexasphere repository (examples/hat_trick_frequencies.py).
# Rebuilt only if missing (delete it to regenerate after changing the Hat Trick code).
data/hat_trick.json: data/hat_frequencies.json
	python make_hat_data.py

# Spectre figures are produced in ../spectre/figures; copy them next to the others.
$(addprefix figures/,$(SPFIG)): figures/%.png: ../spectre/figures/%.png
	cp $< $@

# Figures of the comparison are produced by ../compare_trionda_hat.py and ../the_hat_trick_ball.py into ../examples.
$(addprefix figures/,$(EXFIG)): figures/%.png: ../examples/%.png
	cp $< $@
