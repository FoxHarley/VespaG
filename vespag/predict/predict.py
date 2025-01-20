import csv
import os
import warnings
from pathlib import Path

# import matplotlib.pyplot as plt
import numpy as np

import h5py
import numpy as np
import rich.progress as progress
import torch
from Bio import SeqIO
from tqdm.rich import tqdm

from vespag.data.embeddings import Embedder
from vespag.utils import (
    AMINO_ACIDS,
    GEMME_ALPHABET,
    DEFAULT_MODEL_PARAMETERS,
    Mutation,
    SAV,
    compute_mutation_score,
    get_device,
    load_model,
    mask_non_mutations,
    read_mutation_file,
    setup_logger,
)
from vespag.utils.type_hinting import *

# def visualize_saliency_map(saliency_map, save_path,title="Saliency Map"):
#     plt.figure(figsize=(10, 4))
#     plt.imshow(saliency_map, cmap="hot", aspect="auto")
#     plt.colorbar(label="Gradient Magnitude")
#     plt.title(title)
#     plt.xlabel("Input Dimension")
#     plt.ylabel("Sample Index")
#     plt.savefig(save_path, bbox_inches="tight")
#     plt.close()

def generate_saliency_maps(
    fasta_file: Path,
    output_path: Path,
    embedding_file: Path = None,
    mutation_file: Path = None,
    id_map_file: Path = None,
    zero_based_mutations: bool = False,
    embedding_type: EmbeddingType = "esm2"        
) -> None:
    logger = setup_logger()
    warnings.filterwarnings("ignore", message="rich is experimental/alpha")

    output_path = output_path or Path.cwd() / "output"
    if not output_path.exists():
        logger.info(f"Creating output directory {output_path}")
        output_path.mkdir(parents=True)

    device = get_device()
    params = DEFAULT_MODEL_PARAMETERS
    params["embedding_type"] = embedding_type
    model = load_model(**params).eval().to(device, dtype=torch.float)

    sequences = {rec.id: str(rec.seq) for rec in SeqIO.parse(fasta_file, "fasta")}

    # Load or generate embeddings
    if embedding_file:
        logger.info(f"Loading pre-computed embeddings from {embedding_file}")
        embeddings = {
            id: torch.from_numpy(np.array(emb[()], dtype=np.float32))
            for id, emb in tqdm(
                h5py.File(embedding_file).items(),
                desc="Loading embeddings",
                leave=False,
            )
        }
        if id_map_file:
            id_map = {row[0]: row[1] for row in csv.reader(id_map_file.open("r"))}
            for from_id, to_id in id_map.items():
                embeddings[to_id] = embeddings[from_id]
                del embeddings[from_id]

    else:
        logger.info("Generating ESM2 embeddings")
        if "HF_HOME" in os.environ:
            plm_cache_dir = os.environ["HF_HOME"]
        else:
            plm_cache_dir = Path.cwd() / ".esm2_cache"
            plm_cache_dir.mkdir(exist_ok=True)
        embedder = Embedder("facebook/esm2_t36_3B_UR50D", plm_cache_dir)
        embeddings = embedder.embed(sequences)
        embedding_output_path = output_path / "esm2_embeddings.h5"
        logger.info(
            f"Saving generated ESM2 embeddings to {embedding_output_path} for re-use"
        )
        Embedder.save_embeddings(embeddings, embedding_output_path)

    # Load or generate mutational landscape
    if mutation_file:
        logger.info("Parsing mutational landscape")
        mutations_per_protein = read_mutation_file(
            mutation_file, one_indexed=not zero_based_mutations
        )
    else:
        logger.info("Generating mutational landscape")
        mutations_per_protein = {
            protein_id: [
                SAV(i, wildtype_aa, other_aa, not zero_based_mutations)
                for i, wildtype_aa in enumerate(sequence)
                for other_aa in AMINO_ACIDS
                if other_aa != wildtype_aa
            ]
            for protein_id, sequence in tqdm(sequences.items(), leave=False)
        }

    logger.info("Generating Saliency Maps")

    with progress.Progress(
        progress.TextColumn("[progress.description]Generating Saliency Maps"),
        progress.BarColumn(),
        progress.TaskProgressColumn(),
        progress.TimeElapsedColumn(),
        progress.TextColumn("Current protein: {task.description}"),
    ) as pbar, torch.no_grad():
        overall_progress = pbar.add_task(
            "Generating Saliency Maps",
            total=sum([len(mutations) for mutations in mutations_per_protein.values()]),
        )

        # get the predictions for each sequence in the fasta file
        for id, sequence in sequences.items():
            pbar.update(overall_progress, description=id, advance=1)

            num_residues = len(sequence)
            max_mutations = 20
            embedding_dim = embeddings[id].shape[1]
            saliency_map_array = np.zeros((num_residues, max_mutations, embedding_dim))

            with torch.enable_grad():
                # Prepare the embeddings and enable gradients
                embedding = embeddings[id].to(device)
                embedding.requires_grad = True 

                # Forward pass
                y = model(embedding)

                # Compute saliency for a each mutation 
                for mutation in mutations_per_protein[id]:
                    if isinstance(mutation, Mutation):
                        raise NotImplementedError("Saliency map is only supported for the entire mutational landscape")
                    elif not isinstance(mutation, SAV):
                        raise ValueError(f"Invalid mutation type: {type(mutation)}. Expected SAV")
                    
                    residue_index = mutation.position
                    mutation_index = GEMME_ALPHABET.index(mutation.to_aa)

                    target_output = y[residue_index, mutation_index]
                    # Compute gradients w.r.t. one mutation
                    target_output.backward(retain_graph=True) 

                    # From the embeddings, select just the gradients for the residue in question
                    # Because the other embeddings are all zero due to the nature of network architecture, which only considers one residue at a time for the predictions
                    # Absolute value gives us the saliency (magnitude of influence rather than direction)
                    saliency_map = torch.abs(embedding.grad).cpu().numpy()
                    saliency_map_array[residue_index, mutation_index, :] = saliency_map[residue_index, :]

                    # Clear gradients for the next iteration
                    embedding.grad.zero_()

            # store the results in the output folder as .npy files
            output_file = output_path / (id + "_saliency.npy")
            np.save(output_file, saliency_map_array)

        pbar.remove_task(overall_progress)

def generate_predictions(
    fasta_file: Path,
    output_path: Path,
    embedding_file: Path = None,
    mutation_file: Path = None,
    id_map_file: Path = None,
    single_csv: bool = False,
    no_csv: bool = False,
    h5_output: bool = False,
    zero_based_mutations: bool = False,
    transform_scores: bool = False,
    normalize_scores: bool = False,
    embedding_type: EmbeddingType = "esm2",
) -> None:
    logger = setup_logger()
    warnings.filterwarnings("ignore", message="rich is experimental/alpha")

    output_path = output_path or Path.cwd() / "output"
    if not output_path.exists():
        logger.info(f"Creating output directory {output_path}")
        output_path.mkdir(parents=True)

    device = get_device()
    params = DEFAULT_MODEL_PARAMETERS
    params["embedding_type"] = embedding_type
    model = load_model(**params).eval().to(device, dtype=torch.float)

    sequences = {rec.id: str(rec.seq) for rec in SeqIO.parse(fasta_file, "fasta")}

    if embedding_file:
        logger.info(f"Loading pre-computed embeddings from {embedding_file}")
        embeddings = {
            id: torch.from_numpy(np.array(emb[()], dtype=np.float32))
            for id, emb in tqdm(
                h5py.File(embedding_file).items(),
                desc="Loading embeddings",
                leave=False,
            )
        }
        if id_map_file:
            id_map = {row[0]: row[1] for row in csv.reader(id_map_file.open("r"))}
            for from_id, to_id in id_map.items():
                embeddings[to_id] = embeddings[from_id]
                del embeddings[from_id]

    else:
        logger.info("Generating ESM2 embeddings")
        if "HF_HOME" in os.environ:
            plm_cache_dir = os.environ["HF_HOME"]
        else:
            plm_cache_dir = Path.cwd() / ".esm2_cache"
            plm_cache_dir.mkdir(exist_ok=True)
        embedder = Embedder("facebook/esm2_t36_3B_UR50D", plm_cache_dir)
        embeddings = embedder.embed(sequences)
        embedding_output_path = output_path / "esm2_embeddings.h5"
        logger.info(
            f"Saving generated ESM2 embeddings to {embedding_output_path} for re-use"
        )
        Embedder.save_embeddings(embeddings, embedding_output_path)

    if mutation_file:
        logger.info("Parsing mutational landscape")
        mutations_per_protein = read_mutation_file(
            mutation_file, one_indexed=not zero_based_mutations
        )
    else:
        logger.info("Generating mutational landscape")
        mutations_per_protein = {
            protein_id: [
                SAV(i, wildtype_aa, other_aa, not zero_based_mutations)
                for i, wildtype_aa in enumerate(sequence)
                for other_aa in AMINO_ACIDS
                if other_aa != wildtype_aa
            ]
            for protein_id, sequence in tqdm(sequences.items(), leave=False)
        }

    logger.info("Generating predictions")
    vespag_scores = {}
    scores_per_protein = {}
    with progress.Progress(
        progress.TextColumn("[progress.description]Generating predictions"),
        progress.BarColumn(),
        progress.TaskProgressColumn(),
        progress.TimeElapsedColumn(),
        progress.TextColumn("Current protein: {task.description}"),
    ) as pbar, torch.no_grad():
        overall_progress = pbar.add_task(
            "Generating predictions",
            total=sum([len(mutations) for mutations in mutations_per_protein.values()]),
        )
        for id, sequence in sequences.items():
            pbar.update(overall_progress, description=id)
            embedding = embeddings[id].to(device)
            y = model(embedding)
            y = mask_non_mutations(y, sequence)

            scores_per_protein[id] = {
                mutation: compute_mutation_score(
                    y,
                    mutation,
                    pbar=pbar,
                    progress_id=overall_progress,
                    transform=transform_scores,
                    normalize=normalize_scores,
                )
                for mutation in mutations_per_protein[id]
            }
            if h5_output:
                vespag_scores[id] = y.detach().numpy()

        pbar.remove_task(overall_progress)
    if h5_output:
        h5_output_path = output_path / "vespag_scores_all.h5"
        logger.info(f"Serializing predictions to {h5_output_path}")
        with h5py.File(h5_output_path, "w") as f:
            for id, vespag_prediction in tqdm(vespag_scores.items(), leave=False):
                f.create_dataset(id, data=vespag_prediction)

    if not no_csv:
        logger.info("Generating CSV output")
        if not single_csv:
            for protein_id, mutations in tqdm(scores_per_protein.items(), leave=False):
                output_file = output_path / (protein_id + ".csv")
                with output_file.open("w+") as f:
                    f.write("Mutation,VespaG\n")
                    f.writelines(
                        [f"{str(sav)},{score}\n" for sav, score in mutations.items()]
                    )
        else:
            output_file = output_path / "vespag_scores_all.csv"
            with output_file.open("w+") as f:
                f.write("Protein,Mutation,VespaG\n")
                f.writelines(
                    [
                        line
                        for line in tqdm(
                            [
                                f"{protein_id},{str(sav)},{score}\n"
                                for protein_id, mutations in scores_per_protein.items()
                                for sav, score in mutations.items()
                            ],
                            leave=False,
                        )
                    ]
                )