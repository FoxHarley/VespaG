""" Skipts to load different types of files"""
import os
import glob
import pandas as pd
from tqdm import tqdm
from Bio import AlignIO

def load_fasta_file(path):
    """Load a FASTA file."""
    with open(path, 'r') as file:
        lines = file.readlines()
        sequences = []
        for i in range(0, len(lines), 2):
            sequences.append((lines[i].strip()[1:], lines[i + 1].strip()))
    return pd.DataFrame(sequences, columns=["id", "sequence"])

def load_dms_scores_from_folder(path):
    """Load all DMS scores from CSV files in the specified folder."""
    dms_scores_list = []
    # load the files
    for file in tqdm(glob.glob(f"{path}/*.csv"), unit="file", desc="Loading DMS scores"):
        id = os.path.basename(file).split(".")[0]
        df = pd.read_csv(file)
        df['id'] = id
        dms_scores_list.append(df)
    return pd.concat(dms_scores_list, ignore_index=True)

def get_vespag_scores(path):
    """Get the VespaG scores for each residue and mutation from a CSV file."""
    df = pd.read_csv(path)
    df = df.rename(columns={'Mutation': 'mutation', 'VespaG': 'vespag'})
    df['residue'] = df['mutation'].str[0:-1]
    df['residue_index'] = pd.to_numeric(df['residue'].str[1:])
    df = df.sort_values(by='residue_index').reset_index(drop=True)
    return df[['residue', 'mutation', 'vespag']]

def load_vespag_scores_from_folder(vespag_path):
    """Load all VespaG scores from CSV files in the specified folder."""
    vespag_scores_list = []
    for file in tqdm(glob.glob(f"{vespag_path}/*.csv"), unit="file", desc="Loading VespaG scores"):
        id = os.path.basename(file).split(".")[0]
        df = get_vespag_scores(file)
        df["id"] = id
        vespag_scores_list.append(df)
    return pd.concat(vespag_scores_list, ignore_index=True)

def get_gemme_scores(path):
    """ Get the GEMME scores for each residue and mutation from a text file."""
    df = pd.read_csv(path, sep=' ')
    df = df.T
    df.index = pd.to_numeric(df.index.str.replace('V', ''))
    df.columns = df.columns.str.upper()
    df = df.reset_index()
    df = df.rename(columns={'index': 'residue_index', 'Gemme':'gemme'})
    # one row per cell
    df = df.melt(id_vars=['residue_index'], var_name='substitution', value_name='gemme')
    # determine the original residues
    residues = df[df.gemme.isna()][['substitution', 'residue_index']]
    residues = residues.sort_values(by='residue_index')
    residues['residue'] = residues['substitution'] + residues['residue_index'].astype(str)
    residues = residues[['residue', 'residue_index']]
    # combine the residues, mutation and gemme score
    df = df.dropna()
    df = pd.merge(df, residues, on='residue_index')
    df = df.sort_values(by='residue_index')
    df['mutation'] = df['residue'] + df['substitution']
    df = df.reset_index(drop=True)
    return df[['residue', 'mutation', 'gemme']]

def load_gemme_scores_from_folder(gemme_path):
    """Load all GEMME scores from text files in the specified folder."""
    folders = [f for f in os.listdir(gemme_path) if os.path.isdir(os.path.join(gemme_path, f))]
    gemme_scores_list = []

    for folder in tqdm(folders, unit="folder", desc="Loading Gemme scores"):
        file_path = os.path.join(gemme_path, folder, f"{folder}_normPred_evolCombi.txt")
        df = get_gemme_scores(file_path)
        df['id'] = folder
        gemme_scores_list.append(df)

    gemme_scores = pd.concat(gemme_scores_list, ignore_index=True)
    return gemme_scores

def load_phy_file(path, format="phylip-relaxed", logger=None):
    """Load a phylogenetic tree."""
    alignment = AlignIO.read(path, format)

    if logger:
        logger.info(f"Number of sequences: {len(alignment)}")
        logger.info(f"Aligned sequence length: {alignment.get_alignment_length()}")
    else:
        print(f"Number of sequences: {len(alignment)}")
        print(f"Aligned sequence length: {alignment.get_alignment_length()}")

    data = [(record.id, str(record.seq)) for record in alignment]

    pla2_alignment = pd.DataFrame(data, columns=["id", "aligned_sequence"])
    return pla2_alignment