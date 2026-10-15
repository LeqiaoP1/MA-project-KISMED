# Makefile
.PHONY: presentation
presentation:
	python presentation/build_deck.py \
		--template presentation/Vorlage.pptx \
		--spec presentation/slides_spec.yaml
