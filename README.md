# Gut Microbiome Neurological Disease State Classification

Important files:

1. `1_combine-meta2db-data.py`: This is the script used to combine all the Meta2DB studies' data into a (Sample x Genus Features) matrix. Contains some metadata related to disease category, health status, age, and sex.
2. `meta2db_combined_fecal_abundances.parquet`: This is the output of the above script. This will be used downstream for preprocessing and model building.
3. `2_preprocess-neurological-data.py`: This is what should be used as the foundation for building your deep learning model. It already corrects for study batch effect, does transformation, train/test split, scaling. You should copy this file and add your own model implementation on the bottom.


## Makefile Targets Information

- `environment`: Create conda environment. You can activate this with `conda activate gut-microbiota-pulmonary`.

- `update`: Update existing conda environment if needed.

- `combine`: Combine all the Meta2DB studies' genus-level fecal microbiome data together, into the file `meta2db_combined_fecal_abundances.parquet`.