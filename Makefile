environment:
	conda env create -f environment.yml

update:
	conda env update -n gut-microbiota-pulmonary -f environment.yml --prune

# combine all the Meta2DB studies' genus-level fecal microbiome data together
combine:
	python 1_combine-meta2db-data.py

# to be continued once we have a more finalized implementation

