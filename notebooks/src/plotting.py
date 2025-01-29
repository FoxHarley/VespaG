import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from src.constants import OPTIONS, GEMME_ALPHABET

def plot_mutation_saliency_over_sequence(saliency_df, sequence):
    """ Plots the average mutation effect and saliency for each residue in the sequence """
    # take the average of the mutation effect and saliency for each residue
    avg_mutation_saliency = saliency_df.groupby('residue_index', as_index=False)[['mutation_effect', 'avg_saliency']].mean()

    # normalize mutation and saliency values to [0,1]
    avg_mutation_saliency['normalized_mutation_effect'] = (avg_mutation_saliency['mutation_effect'] - np.min(avg_mutation_saliency['mutation_effect'])) / (np.max(avg_mutation_saliency['mutation_effect']) - np.min(avg_mutation_saliency['mutation_effect']))
    avg_mutation_saliency['normalized_saliency'] = (avg_mutation_saliency['avg_saliency'] - np.min(avg_mutation_saliency['avg_saliency'])) / (np.max(avg_mutation_saliency['avg_saliency']) - np.min(avg_mutation_saliency['avg_saliency']))

    # make two heatmaps for the normalized mutation effect and saliency
    fig, axs = plt.subplots(2, 1, figsize=(20, 4))
    saliency_plot = sns.heatmap(
        avg_mutation_saliency[['normalized_saliency']].T,
        ax=axs[0],
        yticklabels=False,
        xticklabels=list(sequence),
        cmap='inferno',
        cbar=True
    )

    axs[0].set_title("Average Saliency")
    axs[0].set_xticks(range(len(sequence)))
    axs[0].set_xticklabels(list(sequence), rotation=90)

    colorbar_saliency = saliency_plot.collections[0].colorbar
    colorbar_saliency.set_label('Average Saliency', rotation=270, labelpad=20)

    mutation_plot  = sns.heatmap(
        avg_mutation_saliency[['normalized_mutation_effect']].T,
        ax=axs[1],
        yticklabels=False,
        xticklabels=list(sequence),
        cmap='inferno',
        cbar=True
    )

    axs[1].set_title("Average Mutation Effect")
    axs[1].set_xticks(range(len(sequence)))
    axs[1].set_xticklabels(list(sequence), rotation=90)

    colorbar_mutation = mutation_plot.collections[0].colorbar
    colorbar_mutation.set_label('Average Mutation Effect', rotation=270, labelpad=20)

    plt.tight_layout()

    plt.show()

def plot_residue_saliency_3d(saliency_map, residue_index_zero_index, title="3D Saliency Map"):
    """ Plots the saliency for one residue in a 3D plane """
    # check assumptions
    assert saliency_map.shape[1] == 20 and saliency_map.shape[2] == 2560
    slice_saliency = saliency_map[residue_index_zero_index]  # Shape: (20, 2560)

    # normalize saliency values to [0,1]
    slice_saliency = (slice_saliency - np.min(slice_saliency)) / (np.max(slice_saliency) - np.min(slice_saliency))

    # Create grid for 3D surface
    x = np.arange(slice_saliency.shape[0])  # Mutation index (20 amino acids)
    y = np.arange(slice_saliency.shape[1])  # Embedding dimensions (2560)
    x, y = np.meshgrid(x, y)
    z = slice_saliency.T  # Transpose to match grid shape

    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, cmap='inferno')

    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel("Mutation")
    ax.set_ylabel("Embedding Dimension (2560)")
    ax.set_zlabel("Saliency Value")
    ax.set_xticks(np.arange(len(GEMME_ALPHABET)))  # Numeric positions
    ax.set_xticklabels(GEMME_ALPHABET)  # Corresponding amino acid labels

    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, label="Saliency")
    plt.show()

def plot_residue_saliency_2d(saliency_map, residue_index_zero_index, title):
    """ Plots a heatmap with the saliencies for one wildtype """
    # check assumptions
    assert saliency_map.shape[1] == 20 and saliency_map.shape[2] == 2560
    slice_saliency = saliency_map[residue_index_zero_index] 

    # normalize saliency values to [0,1]
    slice_saliency = (slice_saliency - np.min(slice_saliency)) / (np.max(slice_saliency) - np.min(slice_saliency))

    plt.figure(figsize=(12, 6))
    sns.heatmap(slice_saliency, cmap='inferno', cbar=True)
    plt.title(title)
    plt.xlabel("Embedding Dimension (2560)")
    plt.ylabel("Mutation")

    # Change y-axis ticks to GEMME_ALPHABET
    plt.yticks(ticks=np.arange(len(GEMME_ALPHABET)), labels=list(GEMME_ALPHABET))
    plt.show()

def plot_mean_saliency_for_residue(saliency_map, residue_index_zero_index, title):
    """ Plots the mean and the sem of the saliencies for one residue for each embedding dimension """
    # check assumptions
    assert saliency_map.shape[1] == 20 and saliency_map.shape[2] == 2560
    slice_saliency = saliency_map[residue_index_zero_index]  # Shape: (20, 2560)

    # calculate mean saliency for each embedding dimension
    mean_saliency_per_dim = np.mean(slice_saliency, axis=0)
    # calculate the standard error of the mean for each embedding dimension
    sem_saliency_per_dim = np.std(slice_saliency, axis=0) / np.sqrt(slice_saliency.shape[0])

    # plot the mean saliency for each embedding dimension
    plt.figure(figsize=(12, 6))
    plt.plot(mean_saliency_per_dim, label="Mean")
    plt.fill_between(
        np.arange(len(mean_saliency_per_dim)),
        mean_saliency_per_dim - 1.96 * sem_saliency_per_dim,
        mean_saliency_per_dim + 1.96 * sem_saliency_per_dim,
        color="blue",
        alpha=0.2,
        label="±1.96 SE"
    )

    plt.xlabel("Embedding Dimension (2560)")
    plt.ylabel("Mean Saliency")
    plt.legend()
    plt.title(title)
    plt.show()

def plot_mean_saliency_for_sequence(saliency_map, title):
    """ Plots the mean and the sem of the saliencies for one sequence for each embedding dimension """
    # check assumptions
    assert saliency_map.shape[1] == 20 and saliency_map.shape[2] == 2560

    # calculate mean saliency for each embedding dimension
    mean_saliency_per_dim = np.mean(saliency_map, axis=(0, 1))
    assert mean_saliency_per_dim.shape == (2560,)
    # calculate the standard error of the mean for each embedding dimension
    sem_saliency_per_dim = np.std(saliency_map, axis=(0, 1)) / np.sqrt(saliency_map.shape[0] * saliency_map.shape[1])

    # plot the mean saliency for each embedding dimension
    plt.figure(figsize=(12, 6))
    plt.plot(mean_saliency_per_dim, label="Mean")
    plt.fill_between(
        np.arange(len(mean_saliency_per_dim)),
        mean_saliency_per_dim - 1.96 * sem_saliency_per_dim,
        mean_saliency_per_dim + 1.96 * sem_saliency_per_dim,
        color="blue",
        alpha=0.2,
        label="±1.96 SE"
    )

    plt.xlabel("Embedding Dimension (2560)")
    plt.ylabel("Mean Saliency")
    plt.legend()
    plt.title(title)
    plt.show()

def plot_mutation_effect_for_residue(saliency_df, residue_index_one_index, title="Mutation Effect for Residue", color_by=None):
    """ Plots the mutation effect against the mutant for a specific reisude. 
    
    Allows colloration of the scatter plot by saliency scores or amino acid properties.
    Options for color_by: None,
                            'saliency',
                            'mutant_hydropathy_classes',
                            'mutant_volume_classes',
                            'mutant_chemical_classes',
                            'mutant_physicochemical_classes',
                            'mutant_charge_classes',
                            'mutant_polarity_classes',
                            'mutant_hydrogen_donor_acceptor_classes',
                            'change_hydropathy_classes',
                            'change_volume_classes',
                            'change_chemical_classes',
                            'change_physicochemical_classes',
                            'change_charge_classes',
                            'change_polarity_classes',
                            'change_hydrogen_donor_acceptor_classes'    
    """
    # check assumptions
    assert isinstance(saliency_df, pd.DataFrame)

    # filter the dataframe for the specific residue
    residue_df = saliency_df[saliency_df["residue_index"] == residue_index_one_index].copy()

    # normalize the mutation effect to [0,1]
    residue_df['normalized_mutation_effect'] = (residue_df['mutation_effect'] - np.min(residue_df['mutation_effect'])) / (np.max(residue_df['mutation_effect']) - np.min(residue_df['mutation_effect']))

    # define hue according to color by object 
    options = [f'mutant_{option}' for option in OPTIONS] + [f'wildtype_{option}' for option in OPTIONS] + [f'change_{option}' for option in OPTIONS]
    if not color_by:
        hue = None
        palette = None
    elif color_by == "saliency":
        residue_df['normalized_saliency'] = (residue_df['avg_saliency'] - np.min(residue_df['avg_saliency'])) / (np.max(residue_df['avg_saliency']) - np.min(residue_df['avg_saliency']))
        hue = "normalized_saliency"
        palette = "inferno"
    elif color_by in options:
        hue = color_by
        palette = "tab20"
    
    # define ordering according to order by object
    order_col = "avg_saliency" if color_by == "saliency" else "normalized_mutation_effect"
    residue_df = residue_df.sort_values(by=order_col, ascending=False)

    # plot the mutation effect against the mutant
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x="mutant", y="normalized_mutation_effect", data=residue_df, hue=hue if hue else None, palette=palette)
    if hue:
        plt.legend(title=color_by)
    plt.title(title)
    plt.xlabel("Mutation")
    plt.ylabel("Norm. Mutation Effect")
    plt.show()

def plot_interactive_mutant_effect_vs_saliency(saliency_df, title="Mutation Effect vs Saliency", color_by=None):
    """ Generates an interactive scatter plot with the mutation effect vs. the average saliency, colored by selection. 
        Hovering over a data point shows additional details.
    
    Options for color_by: 
        None,
        'wildtype',
        'mutant', 
        'mutant_hydropathy_classes',
        'mutant_volume_classes',
        'mutant_chemical_classes',
        'mutant_physicochemical_classes',
        'mutant_charge_classes',
        'mutant_polarity_classes',
        'mutant_hydrogen_donor_acceptor_classes',
        'change_hydropathy_classes',
        'change_volume_classes',
        'change_chemical_classes',
        'change_physicochemical_classes',
        'change_charge_classes',
        'change_polarity_classes',
        'change_hydrogen_donor_acceptor_classes'
    """
    width = 600
    height = 600
    colorbar_title = "Legend" if color_by else None

    # Check input type
    assert isinstance(saliency_df, pd.DataFrame)

    # Normalize mutation effect and saliency to [0,1]
    plotting_df = saliency_df.copy()
    plotting_df['scaled_mutation_effect'] = (plotting_df['mutation_effect'] - np.min(plotting_df['mutation_effect'])) / (np.max(plotting_df['mutation_effect']) - np.min(plotting_df['mutation_effect']))
    plotting_df['avg_saliency'] = (plotting_df['avg_saliency'] - np.min(plotting_df['avg_saliency'])) / (np.max(plotting_df['avg_saliency']) - np.min(plotting_df['avg_saliency']))

    # Define hover data to show more details
    hover_data = ["mutant", "wildtype", "residue_index", "mutation_effect", "avg_saliency"]

    # Create the scatter plot
    fig = px.scatter(
        plotting_df, 
        x="avg_saliency", 
        y="scaled_mutation_effect", 
        color=color_by if color_by else None,
        hover_data=hover_data,
        color_continuous_scale="Inferno" if color_by == "avg_saliency" else None,
        title=title
    )

    # Fix the scatter plot size (so that legend does not shrink it)
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        xaxis_title="Average Saliency",
        yaxis_title="Mutation Effect",
        xaxis=dict(range=[0,1], fixedrange=True),
        yaxis=dict(range=[0,1], fixedrange=True),
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50)  # Reduce right margin slightly
    )

    # Move legend outside and resize it
    if color_by:
        fig.update_layout(
            legend=dict(
                title=colorbar_title,  # Fix title
                x=1.05,  # Move legend outside plot
                y=1,  # Align legend at the top
                orientation="v"  # Ensure vertical alignment
            ),
            coloraxis_colorbar=dict(
                title=colorbar_title,  # Colorbar title
                thickness=15,  # Slim colorbar
                len=0.6,  # Make colorbar shorter
                xpad=20  # Add spacing between plot and legend
            )
        )

    fig.show()

def plot_parallel_coordinates(saliency_df, columns_to_use, title="Parallel Coordinates Plot"):
    """ Plots a parallel coordinates plot for the saliency dataframe """
    # check assumptions
    assert isinstance(saliency_df, pd.DataFrame)

    plotting_df = saliency_df.copy()

    # Normalize mutation effect and saliency to [0,1]
    plotting_df['mutation_effect'] = (plotting_df['mutation_effect'] - np.min(plotting_df['mutation_effect'])) / (np.max(plotting_df['mutation_effect']) - np.min(plotting_df['mutation_effect']))
    plotting_df['avg_saliency'] = (plotting_df['avg_saliency'] - np.min(plotting_df['avg_saliency'])) / (np.max(plotting_df['avg_saliency']) - np.min(plotting_df['avg_saliency']))

    # Factorize categorical columns for numeric encoding
    label_mapping = {}  
    factorized_columns = []  # Store only transformed columns

    options = [f'mutant_{option}' for option in OPTIONS] + [f'wildtype_{option}' for option in OPTIONS] + [f'change_{option}' for option in OPTIONS] + ['mutant', 'wildtype']
    for column in columns_to_use:
        if column in options:  # Factorize only categorical columns
            plotting_df[column + '_factorized'], unique_labels = pd.factorize(plotting_df[column], sort=True)
            label_mapping[column] = dict(enumerate(unique_labels))
            factorized_columns.append(column + '_factorized')  # Store transformed column
    # Print label mapping
    print("Label Mapping:")
    for column, mapping in label_mapping.items():
        print(f"{column}: {mapping}")
    
    # Create the parallel coordinates plot
    fig = px.parallel_coordinates(
        plotting_df, 
        dimensions=['avg_saliency'] + factorized_columns + ['mutation_effect'],  # Use correct columns
        color="mutation_effect",  # Coloring by mutation effect
        labels={
            "mutation_effect": "Mutation Effect",
            "avg_saliency": "Avg. Saliency",
            **{col + '_factorized': col.replace("_classes", "").replace("_factorized", "") for col in columns_to_use if col in saliency_df.columns}
        },
        color_continuous_scale="Bluered",
        title=title
    )

    fig.show()    

def plot_distribution(saliency_df, data_column, label_column, title, color_by=None):
    """ Plots the distribution of the provided column using boxplot and violin plot
    
    Offers a colorisation per amino acid group. 
    """
    # check assumptions
    assert isinstance(saliency_df, pd.DataFrame)

    # pick plotting data 
    selected_columns = list({data_column, label_column, color_by} - {None})
    plotting_data = saliency_df[selected_columns].copy()

    # normalize the data column 
    plotting_data['normalized_data'] = (plotting_data[data_column] - np.min(plotting_data[data_column])) / (np.max(plotting_data[data_column]) - np.min(plotting_data[data_column]))

    # order by column 
    ordered_index = plotting_data.groupby(label_column)['normalized_data'].median().sort_values(ascending=False).index.tolist()

    plt.figure(figsize=(10, 5))
    
    # boxplot (adds the legend)
    sns.boxplot(
        x=label_column,
        y='normalized_data',
        data=plotting_data,
        palette="tab20" if color_by else None,
        boxprops=dict(alpha=.5),
        hue=color_by if color_by else None,
        order=ordered_index
    )

    # violin plot (does NOT add the legend)
    sns.violinplot(
        x=label_column,
        y='normalized_data',
        data=plotting_data,
        inner=None, 
        cut=0,
        alpha=0.7,
        palette="tab20" if color_by else None,
        hue=color_by if color_by else None,
        order=ordered_index,
        legend=False  # Prevents duplicate legend
    )

    plt.title(title)
    plt.xlabel(label_column)
    plt.ylabel(data_column)

    # Add legend manually if color_by is set
    if color_by:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # Remove duplicates
        plt.legend(by_label.values(), by_label.keys(), title=color_by.split('_classes')[0], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.show()
