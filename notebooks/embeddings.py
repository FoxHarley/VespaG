import numpy as np
import h5py
import seaborn as sns
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

embedding_file = '/mnt/f/nicole/embeddings/pla2_esm2_embeddings.h5'

embeddings = []

# load the entire h5 file 
with h5py.File(embedding_file, 'r') as f:
    for embedding in f.values():
        embeddings.append(embedding[:])

# concatenate all the embeddings
embeddings = np.concatenate(embeddings, axis=0)
# flatten the embeddings
flattened_embeddings = embeddings.flatten()

# get the mean and sem of the embeddings
mean = np.mean(flattened_embeddings)
sem = np.std(flattened_embeddings) / np.sqrt(len(flattened_embeddings))
min = np.min(flattened_embeddings)
max = np.max(flattened_embeddings)
count = len(flattened_embeddings)
q25 = np.percentile(flattened_embeddings, 25)
q50 = np.percentile(flattened_embeddings, 50) 
q75 = np.percentile(flattened_embeddings, 75)

print('Statistics for all embeddings:')
print(f"Mean: {mean:.6f} ± {sem:.6f}")
print(f"Min: {min:.6f}")
print(f"25th percentile: {q25:.6f}")
print(f"50th percentile: {q50:.6f}")
print(f"75th percentile: {q75:.6f}")
print(f"Max: {max:.6f}")
print(f"Count: {count}")
print()

# select all embeddings except for 1542
reduced_embeddings = embeddings[:, np.arange(embeddings.shape[1]) != 1542]
# flatten the reduced embeddings
flattened_reduced_embeddings = reduced_embeddings.flatten()

# get the mean and sem of the reduced embeddings
mean = np.mean(flattened_reduced_embeddings)
sem = np.std(flattened_reduced_embeddings) / np.sqrt(len(flattened_reduced_embeddings))
min = np.min(flattened_reduced_embeddings)
max = np.max(flattened_reduced_embeddings)
count = len(flattened_reduced_embeddings)
q25 = np.percentile(flattened_reduced_embeddings, 25)
q50 = np.percentile(flattened_reduced_embeddings, 50)
q75 = np.percentile(flattened_reduced_embeddings, 75)
print('Stats for all embeddings except for 1542')
print(f"Mean: {mean:.6f} ± {sem:.6f}")
print(f"Min: {min:.6f}")
print(f"25th percentile: {q25:.6f}")
print(f"50th percentile: {q50:.6f}")
print(f"75th percentile: {q75:.6f}")
print(f"Max: {max:.6f}")
print(f"Count: {count}")
print()

# get the same statistics for the 1542 dimension
dimension = embeddings[:, 1542]
mean = np.mean(dimension)
sem = np.std(dimension) / np.sqrt(len(dimension))
min = np.min(dimension)
max = np.max(dimension)
count = len(dimension)
q25 = np.percentile(dimension, 25)
q50 = np.percentile(dimension, 50)
q75 = np.percentile(dimension, 75)
print('Statistics for dimension 1542')
print(f"Mean: {mean:.6f} ± {sem:.6f}")
print(f"Min: {min:.6f}")
print(f"25th percentile: {q25:.6f}")
print(f"50th percentile: {q50:.6f}")
print(f"75th percentile: {q75:.6f}")
print(f"Max: {max:.6f}")
print(f"Count: {count}")
print()