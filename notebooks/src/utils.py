import numpy as np 

GEMME_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

def highest_scoring_saliencies(saliency_map, residue_index_zero_index, n):
    """ Determins the n highest scoring saliencies for a specific residue and returns a list with tuples [(Mutant, Embedding Dimension)] """
    # check assumptions 
    assert saliency_map.shape[1] == 20 and saliency_map.shape[2] == 2560
    saliency_slice = saliency_map[residue_index_zero_index]  # Shape: (20, 2560)
    
    # Flatten the 2D array into a 1D array
    flat_indices = np.argsort(saliency_slice.ravel())[-n:][::-1]  # Get indices of top N values

    # Convert flat indices back to 2D indices
    mutant_indices, embedding_dims = np.unravel_index(flat_indices, saliency_slice.shape)

    return [(GEMME_ALPHABET[mutant_indices[x]], embedding_dims[x]) for x in range(n)]