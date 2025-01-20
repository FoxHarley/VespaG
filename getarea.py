""" Scripts for parsing getarea files """
import pandas as pd
import re
import os

# Mapping of three-letter amino acid codes to one-letter codes
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E',
    'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V'
}

def parse_getarea_file(filepath):
    ''' In the format given by the suppementary information '''

    # Define the column names
    columns = ["Residue", "Position", "Total", "Apolar", "Backbone", "Sidechain", "Ratio", "In/Out"]
    
    # Initialize an empty list to store the rows of data
    data = []
    
    # Open and read the file
    with open(filepath, 'r') as file:
        for line in file:
            # Use regex to match the expected line format for data rows
            match = re.match(r'\s*([A-Z]{3})\s+(\d+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+(\d+\.\d+|\d+)?\s*(\w?)', line)
            if match:
                # Extract the matched groups to form a row of data
                residue, position, total, apolar, backbone, sidechain, ratio, in_out = match.groups()
                in_out = in_out if in_out else None  # Set to NaN if no value is present
                row = [THREE_TO_ONE[residue], position, float(total), float(apolar), float(backbone), float(sidechain), float(ratio), in_out]
                data.append(row)
    
    # Create a DataFrame from the collected data
    df = pd.DataFrame(data, columns=columns)
    return df

def parse_getarea_output(file_path):
    with open(file_path, 'r', encoding='iso-8859-1') as file:
        content = file.read()

    # Update the pattern to also match NaN or empty values
    pattern = r"(\w{3,4})\s+(\d+)\s+([\d\.]+|NaN)\s+([\d\.]+|NaN)\s+([\d\.]+|NaN)\s+([\d\.]+|NaN)\s+([\d\.]+|NaN)\s+(\w?)"
    matches = re.findall(pattern, content)

    # Prepare lists to store the extracted data
    residues, residue_numbers, totals, apolars, backbones, sidechains, ratios, in_outs = [], [], [], [], [], [], [], []
    
    # Process each match and append to respective lists
    for match in matches:
        residue, residue_number, total, apolar, backbone, sidechain, ratio, in_out = match
        residues.append(THREE_TO_ONE[residue])
        residue_numbers.append(int(residue_number))
        totals.append(float(total) if total != 'NaN' else None)
        apolars.append(float(apolar) if apolar != 'NaN' else None)
        backbones.append(float(backbone) if backbone != 'NaN' else None)
        sidechains.append(float(sidechain) if sidechain != 'NaN' else None)
        ratios.append(float(ratio) if ratio != 'NaN' else None)
        in_outs.append(in_out if in_out else None)

    # Create DataFrame with extracted data
    df = pd.DataFrame({
        'residue': residues,
        'residue_number': residue_numbers,
        'total': totals,
        'apolar': apolars,
        'backbone': backbones,
        'sidechain': sidechains,
        'ratio': ratios,
        'in_out': in_outs
    })

    # Combine 'Residue' and 'Residue Number' columns without the decimal point
    df['residue'] = df['residue'] + df['residue_number'].astype(str)

    return df

def get_getarea_data_from_folder(getarea_path):
    ''' Get all getarea files from a folder '''
    getarea_files = [os.path.join(getarea_path, x) for x in os.listdir(getarea_path)]

    getarea_dfs = []
    for file in getarea_files:
        # extract id from file name
        file_name = file.split('/')[-1].split('.')[0]
        id = file_name.split('_unrelaxed_rank')[0]
        
        # parse the file
        df = parse_getarea_output(file)
        df['id'] = id
        getarea_dfs.append(df)

    getarea_df = pd.concat(getarea_dfs)

    return getarea_df