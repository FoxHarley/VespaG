from pathlib import Path
from typing import Annotated, Optional

import typer

from .data.embeddings import generate_embeddings
from .eval import eval
from .predict import generate_predictions, generate_saliency_maps, generate_guided_saliency_maps
from .training.train import train as run_training
from .utils.type_hinting import EmbeddingType

app = typer.Typer()

app.add_typer(eval.app, name="eval")


@app.command()
def predict(
    fasta_file: Annotated[
        Path,
        typer.Option(
            "-i",
            "--input",
            help="Path to FASTA-formatted file containing protein sequence(s)",
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output",
            help="Path for saving created CSV and/or H5 files. Defaults to ./output",
        ),
    ] = None,
    embedding_file: Annotated[
        Path,
        typer.Option(
            "-e",
            "--embeddings",
            help="Path to pre-generated input embeddings. Embeddings will be generated from scratch if no path is provided.",
        ),
    ] = None,
    mutation_file: Annotated[
        Path,
        typer.Option(
            "--mutation-file", help="CSV file specifying specific mutations to score"
        ),
    ] = None,
    id_map_file: Annotated[
        Path,
        typer.Option(
            "--id-map",
            help="CSV file mapping embedding IDs to FASTA IDs if they're different",
        ),
    ] = None,
    single_csv: Annotated[
        Optional[bool],
        typer.Option(
            "--single-csv/--multi-csv",
            help="Whether to return one CSV file for all proteins instead of a single file for each protein",
        ),
    ] = False,
    no_csv: Annotated[
        bool,
        typer.Option(
            "--no-csv/--csv", help="Whether no CSV output should be produced at all"
        ),
    ] = False,
    h5_output: Annotated[
        bool,
        typer.Option(
            "--h5-output/--no-h5-output",
            help="Whether a file containing predictions in HDF5 format should be created",
        ),
    ] = False,
    zero_based_mutations: Annotated[
        bool,
        typer.Option(
            "--zero-idx/--one-idx",
            help="Whether to enumerate the sequence starting at 0",
        ),
    ] = False,
    transform_scores: Annotated[
        bool,
        typer.Option(
            "--transform/--dont-transform",
            help="Whether to transform scores to same distribution as GEMME scores",
        ),
    ] = False,
    normalize_scores: Annotated[
        bool,
        typer.Option(
            "--normalize/--dont-normalize",
            help="Whether to transform scores to [0, 1] range",
        ),
    ] = False,
    embedding_type: Annotated[
        EmbeddingType,
        typer.Option(
            "--embedding-type", help="Type of pLM used for generating embeddings"
        ),
    ] = "esm2",
) -> None:
    id_map_file = None
    generate_predictions(
        fasta_file,
        output_path,
        embedding_file,
        mutation_file,
        id_map_file,
        single_csv,
        no_csv,
        h5_output,
        zero_based_mutations,
        transform_scores,
        normalize_scores,
        embedding_type,
    )

@app.command()
def saliency(
    fasta_file: Annotated[
        Path,
        typer.Option(
            "-i",
            "--input",
            help="Path to FASTA-formatted file containing protein sequence(s)",
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output",
            help="Path for saving generated saliency maps. Defaults to ./output",
        ),
    ] = None,
    embedding_file: Annotated[
        Path,
        typer.Option(
            "-e",
            "--embeddings",
            help="Path to pre-generated input embeddings. Embeddings will be generated from scratch if no path is provided.",
        ),
    ] = None,
    mutation_file: Annotated[
        Path,
        typer.Option(
            "--mutation-file", help="CSV file specifying specific mutations to score"
        ),
    ] = None,
    id_map_file: Annotated[
        Path,
        typer.Option(
            "--id-map",
            help="CSV file mapping embedding IDs to FASTA IDs if they're different",
        ),
    ] = None,
    zero_based_mutations: Annotated[
        bool,
        typer.Option(
            "--zero-idx/--one-idx",
            help="Whether to enumerate the sequence starting at 0",
        ),
    ] = False,
    embedding_type: Annotated[
        EmbeddingType,
        typer.Option(
            "--embedding-type", help="Type of pLM used for generating embeddings"
        ),
    ] = "esm2",
) -> None:
    """
    Generate saliency maps for mutational landscapes from input protein sequences.
    """
    generate_saliency_maps(
        fasta_file=fasta_file,
        output_path=output_path,
        embedding_file=embedding_file,
        mutation_file=mutation_file,
        id_map_file=id_map_file,
        zero_based_mutations=zero_based_mutations,
        embedding_type=embedding_type,
    )

@app.command(name="guided_saliency")
def guided_saliency(
    fasta_file: Annotated[
        Path,
        typer.Option(
            "-i",
            "--input",
            help="Path to FASTA-formatted file containing protein sequence(s)",
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output",
            help="Path for saving generated saliency maps. Defaults to ./output",
        ),
    ] = None,
    embedding_file: Annotated[
        Path,
        typer.Option(
            "-e",
            "--embeddings",
            help="Path to pre-generated input embeddings. Embeddings will be generated from scratch if no path is provided.",
        ),
    ] = None,
    mutation_file: Annotated[
        Path,
        typer.Option(
            "--mutation-file", help="CSV file specifying specific mutations to score"
        ),
    ] = None,
    id_map_file: Annotated[
        Path,
        typer.Option(
            "--id-map",
            help="CSV file mapping embedding IDs to FASTA IDs if they're different",
        ),
    ] = None,
    zero_based_mutations: Annotated[
        bool,
        typer.Option(
            "--zero-idx/--one-idx",
            help="Whether to enumerate the sequence starting at 0",
        ),
    ] = False,
    embedding_type: Annotated[
        EmbeddingType,
        typer.Option(
            "--embedding-type", help="Type of pLM used for generating embeddings"
        ),
    ] = "esm2",
) -> None:
    """
    Generate saliency maps for mutational landscapes from input protein sequences using guided backpropagation.
    """
    generate_guided_saliency_maps(
        fasta_file=fasta_file,
        output_path=output_path,
        embedding_file=embedding_file,
        mutation_file=mutation_file,
        id_map_file=id_map_file,
        zero_based_mutations=zero_based_mutations,
        embedding_type=embedding_type,
    )


@app.command()
def embed(
    input_fasta_file: Annotated[Path, typer.Argument(help="Path of input FASTA file")],
    output_h5_file: Annotated[
        Path, typer.Argument(help="Path for saving HDF5 file with computed embeddings")
    ],
    cache_dir: Annotated[
        Path,
        typer.Option(
            "-c", "--cache-dir", help="Custom path to download model checkpoints to"
        ),
    ],
    embedding_type: Annotated[
        EmbeddingType,
        typer.Option(
            "-e",
            "--embedding-type",
            case_sensitive=False,
            help="Type of embeddings to generate",
        ),
    ] = EmbeddingType.esm2,
    pretrained_path: Annotated[
        str,
        typer.Option("--pretrained-path", help="Path or URL of pretrained transformer"),
    ] = None,
):
    generate_embeddings(
        input_fasta_file, output_h5_file, cache_dir, embedding_type, pretrained_path
    )


@app.command()
def train(
    model_config_key: Annotated[str, typer.Option("--model")],
    datasets: Annotated[list[str], typer.Option("--dataset")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")],
    embedding_type: Annotated[str, typer.Option("--embedding-type", "-e")],
    compute_full_train_loss: Annotated[bool, typer.Option("--full-train-loss")] = False,
    sampling_strategy: Annotated[str, typer.Option("--sampling-strategy")] = "basic",
    wandb_config: Annotated[tuple[str, str], typer.Option("--wandb")] = None,
    limit_cache: Annotated[bool, typer.Option("--limit-cache")] = False,
    use_full_dataset: Annotated[bool, typer.Option("--use-full-dataset")] = False,
):
    run_training(
        model_config_key,
        datasets,
        output_dir,
        embedding_type,
        compute_full_train_loss,
        sampling_strategy,
        wandb_config,
        limit_cache,
        use_full_dataset,
    )


if __name__ == "__main__":
    app()
