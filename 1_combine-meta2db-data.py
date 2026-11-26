"""
Script to combine all the Meta2DB studies' genus-level data specific to the gut microbiome.

IMPORTANT: 
you don't need to run this script again if you want to look at only genus-level data.
just use the output file 'meta2db_combined_fecal_abundances.parquet' downstream,
start from '2_preprocess-neurological-data.py'.

Meta2DB metadata and microbiome profiles can be downloaded at either of these links:
- https://zenodo.org/records/17315984
- https://gdo-meta2db.llnl.gov/downloads
"""

from pathlib import Path
import pandas as pd

# read in metadata - fix path as needed
meta = pd.read_csv('../meta2db/meta2db_metadata.csv', encoding='cp1252')

# take only the fecal samples
fecal_meta = meta[meta['host_body_product'] == 'feces'].copy()
print(f"Shape of fecal sample metadata: {meta.shape}")

# rename some metadata study names to more programmatically find those studies later
fecal_meta['project_name'] = fecal_meta['project_name'].replace('iHMP_IBD', '2022_IHMP')

meta_cols = ['project_name', 'disease_category', 'health_disease_stat', 'host_age', 'host_sex']
all_abundances = []
all_metas = []

# directory with all the Meta2DB microbiome abundance files
dir = Path("../meta2db/meta2db_microbiome profiles")

for file in dir.iterdir():
    # split by underscore first
    split_study_underscore = file.name.split("_")[0:2]
    study_name = split_study_underscore[0] + "_" + split_study_underscore[1]

    if "-" in study_name:
        # need to split by dash secondly to match ALL the study names
        split_study_dash = study_name.split("-")[0:2]
        study_name = split_study_dash[0] + "-" + split_study_dash[1]

    # match the current study name to its corresponding rows in metadata
    matches = fecal_meta[fecal_meta['project_name'].str.contains(study_name)]

    if len(matches) == 0:
        print(f"Study {study_name} did not match any in fecal metadata")
        continue

    # read in abundance data - it has a multi-index header
    df = pd.read_csv(file, header=[0,1], index_col=0)

    # filter to only take a certain taxa level of data
    taxa_level = 'genus'
    df = df.loc[df[('Details', 'Rank')] == taxa_level, :]

    # set the index to be the genus name instead
    df = df.groupby(('Details', 'Name')).sum(numeric_only=True) # sum up any duplicates' abundances
    df.index.name = taxa_level

    # take only the 'cnt' column containing the measured microbiota counts
    cnt_cols = df.columns.get_level_values(1) == 'cnt' # on level 1 of the multi-index header
    df_cnt = df.loc[:,cnt_cols].copy()
    df_cnt.columns = df_cnt.columns.droplevel('Stats')

    # get df to be in Samples x Taxa, and get matching name with the metadata
    df_samples = df_cnt.T
    df_samples.index = df_samples.index.str.split('_').str[0]
    df_samples.index.name = 'run_acc' # sample accession number

    # match up the metadata cols with the abundance data
    meta_match_samples = fecal_meta[fecal_meta['run_acc'].isin(df_samples.index)].copy()
    meta_match_samples = meta_match_samples.drop_duplicates(subset='run_acc').set_index('run_acc')[meta_cols]

    # align the matched rows and store counts in compressed format - in case DF becomes too big
    aligned_df = df_samples.loc[meta_match_samples.index].fillna(0).astype(int)
    compressed_df = aligned_df.astype(pd.SparseDtype("int", fill_value=0))

    # add it to a list of all study dataframes we will combine together later
    all_metas.append(meta_match_samples)
    all_abundances.append(compressed_df)
    print(f"Done processing study {study_name}...")

### combine all the studies' dataframes together 
# metadata
combined_meta = pd.concat(all_metas, axis=0)
print("Done combining all metadata together")

# abundances
combined_abundances = pd.concat(all_abundances, axis=0)
print("Done combining all abundance data together")

# convert sprase -> dense so we can save to parquet file, then fill NAs with 0s
combined_abundances = combined_abundances.sparse.to_dense().fillna(0)

# filter out features that have super small variances
combined_abundances = combined_abundances.loc[:, combined_abundances.var() > 0.01]
print("Filtered out features that have variances at 0.01 or less")

# final dataframe that will have meta + abundance together
final_df = pd.concat([combined_meta, combined_abundances], axis=1)
print("Done combining the metadata and the combined abundances together")

# save to file
final_df.to_parquet("meta2db_combined_fecal_abundances.parquet", engine="pyarrow")
print("Finished saving to parquet...")

# final_df.to_csv("meta2db_combined_fecal_abundances.csv")
# print("Finished saving to csv...")

print(f"{final_df.shape[0]} x {final_df.shape[1]} combined dataframe saved to file.")