# Hoplite

Hoplite is a database for working with large numbers of embeddings.

We develop it with bioacoustic audio applications in mind, where we have a large number of audio files and (usually) a small number of human annotations on those files. Each audio file is typically split up into (possibly overlapping) audio windows, and we have an embedding for each audio window.

Structurally, however, the database is not limited to audio, and may be useful for other domains (image embeddings, satellite image embeddings, etc).

The database handles the embeddings, source metadata for each embedding, labels (annotations) for individual embeddings, and a graph structure (edges) on the embeddings (used for fast approximate nearest neighbor search).

## Backends

The Hoplite database is accessed via the API in `interface.py`. It is possible to reimplement the API on different backends. As of this writing, two backends are implemented: an SQLite database and an in-memory implementation which simply uses a number of Numpy arrays.

The `db_loader.py` file contains some tooling for dumping an instance of a Hoplite database into a new backend. (eg, persisting an in-memory database to SQLite, or loading an SQLite database into memory for faster operation.)

## Database structure

The tables within the database are as follows:

The **embeddings** table contains the embedding vectors, produced by some pretrained model. Each embedding has a unique ID, as well as a *source* and *offset* within the source. The offset is an arbitrary vector, indicating what region of the source the embedding refers to. For example, when working with bioacoustic audio and embeddings of fixed-length audio windows, the offset can be simply the time offset within the source audio file for the embedded audio. For image applications, the offset could be empty (indicating the whole image), or give the corners of a bounding box.

The **source** table holds information for joining embeddings with the source data.  Each *source* consists of a unique ID and a string identifier, for example a relative file path under some root directory, and a *dataset name*, which allows defining groups of related sources.

The **labels** table contains any labels or annotations on the embeddings. Each label consists of an *embedding id*, the *label* string, a *type* (eg, positive or negative), and a *provenance* string (eg, annotator's name, or id for the model which produced a pseudo-label). You can have many labels per embedding, for example indicating different entities represented in the audio, or different annotator's decisions for the same label+embedding pair.

The **edges** table is used for fast approximate nearest neighbor embeddings search.

Finally, the **metadata** table is a string-to-json key-value store for handling arbitrary additional metadata. This is used, for example, to indicate which embedding model was used to produce the embeddings in the database.

## Usage Examples

For up-to-date usage, see `tests/hoplite_test.py`. We give a brief overview of functionality here, however.

### Creating a Database

The in-memory database uses fixed-size numpy arrays for storing embeddings and sparse lists of graph edges. The embedding dimension should be known from the embedding model. The `max_size` is the maximum number of embeddings the database will contain. The `degree_bound` determines the maximum number of outgoing edges per embedding.

```
from chirp.projects.hoplite import in_mem_impl

db = in_mem_impl.InMemoryGraphSearchDB.create(
    embedding_dim=1280,
    max_size=2000,
    degree_bound=256,
)
db.setup()
```

The SQLite backend has no particular limit on the number of embeddings or number of edges per vertex. You can create an SQLite backend like so:

```
from chirp.projects.hoplite import sqlite_impl

db = sqlite_impl.SQLiteGraphSearchDB.create(
    db_path=db_file_path,
    embedding_dim=1280,
)
db.setup()
```

### Working with Embeddings

Adding embeddings to the database is simple. Describe the embedding source, and insert it using the API. The insertion call will return the unique ID for this embedding.

```
from chirp.projects.hoplite import interface

dataset_name = 'my_data'
source_id = 'some_file.wav'
offsets = np.array([100.0])
embedding = np.random.normal(size=[1280])

source = interface.EmbeddingSource(dataset_name, source_id, offsets)
uid = db.insert_embedding(embedding, source)
```

We can retrieve collections of embeddings with an array of UIDs. Note that backends with threading (eg, SQLite) may shuffle the order of the embeddings returned, so `get_embeddings` returns both the possibly-shuffled UIDs and associated embeddings in the matching order.
```
uids = np.array([uid1, uid2])
uids, embeddings = db.get_embeddings(uids)
```

### Working with Labels

Now to add a couple labels to the embedding:

```
label = interface.Label(
  embedding_id=uid,
  label='birb',
  type=interface.LabelType.POSITIVE,
  provenance='me')
db.insert_label(label)

other_label = interface.Label(
  embedding_id=uid,
  label='birb',
  type=interface.LabelType.NEGATIVE,
  provenance='you')
db.insert_label(other_label)
```

We can now get all the labels assigned to the embedding:
```
labels = db.get_labels(uid)
```

To retrieve embeddings with particular labels, we use `get_embeddings_by_label`.
By default, we get all POSITIVE embeddings annotated with the target label string, regardless of provenance.

```
got = db.get_embeddings_by_label('birb')
```

We can add `type` and `provenance` arguments to select embeddings with a particular label type or provenance, however. Specifying `None` for type or provenance will give unconstrained results.

```
got = db.get_embeddings_by_label('birb', provenance='me')
got = db.get_embeddings_by_label('birb', type=interface.LabelType.POSITIVE)

# Returns all embeddings with POSITIVE or NEGATIVE labels for 'birb'.
got = db.get_embeddings_by_label('birb', type=None)
```

### Working with Sources

To get all embeddings with a particular `dataset_name`:
```
got = db.get_embeddings_by_source(dataset_name='my_data')
```

To get all embeddings form a particular source:
```
got = db.get_embeddings_by_source(
  dataset_name='my_data', source_id='some_file.wav')
```

And to find out the source for some particular embedding:
```
source = db.get_embedding_source(uid)
```

### Utility functions

Some core info that we typically need to refer to can be accessed through the API:

```
embedding_count = db.count_embeddings()

# Number of distinct labels in the database.
classes_count = db.count_classes()
```

We can get a random (in the [xkcd](https://xkcd.com/221/) sense) embedding ID from the database:
```
uid = db.get_one_embedding_id()
```

## Vector Similarity Search

We provide utilities for both brute force and indexed vector search.

Vector search requires a 'scoring function.' In this library, higher scores must indicate greater similarity. So, good scoring functions include cosine similarity, inner product, or *negative* Euclidean distance (nearest neighbor).

For bioacoustic embeddings, we recommend *inner product*, as it handles mixture embeddings for overlapping calls more gracefully than Euclidean distance or cosine similarity. However, the user is invited to experiment!

### Brute-force search

Brute force search capabilities are handled by `brutalism.py`. Here's some example usage:

```
from chirp.projects.hoplite import brutalism

query = np.random.normal(size=[1280])
score_fn = np.dot
results, scores = brutalism.threaded_brute_search(db, query, score_fn)

# The `results` are a search_results.TopKSearchResults object.
# This wraps a list of search results kept in heap order, but iterating over the
# TopKSearchResults object will yield results in descending order.
for r in results:
  print(r.embedding_id, r.sort_score)
```

Brute force search is appropriate in the following situations:

* Perfect recall is needed.
* The dataset contains less than about a million embeddings.
* A flexible scoring function is needed (eg, for margin sampling).

We can also use sampled search to apply brute force search on a random subset of data:

```
# Find elements whose dot product with q is approximately 2.
weird_score_fn = lambda q, t: -np.abs(2.0 - np.dot(q, t))
results, scores = brutalism.threaded_brute_search(
  db, query, weird_score_fn,
  sample_size=0.1)  # Search a random 10% of the embeddings.

results, scores = brutalism.threaded_brute_search(
  db, query, weird_score_fn,
  sample_size=100_000)  # Search up to 100k randomly selected embeddings.
```

In short, you should only move to indexed search when you're really annoyed with how slow brute force search is, don't mind missing some results, and have a specific scoring function to work with.

### Indexed Vector Search

*Under Construction*

For very large datasets, approximate nearest neighbor search can greatly speed up analysis. For this, we define a graph on the embeddings and explore using a greedy nearest-neighbor search. The algorithm is an adaptation of the Vamana algorithm, detailed in the [DiskANN paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf).

Indexing is performed with algorithms implemented in JAX, which can run on either CPU or GPU. GPU indexing is considerably faster than CPU, and likely necessary for datasets at the scale where indexing is necessary. However, GPUs have limited memory. Thus, we implement a *sharded* indexing strategy, where subsets of the data are indexed, and then those shard indexes are merged into a single large index, which is stored in the DB.

For now, JAX indexing only uses the inner product metric.

```
from chirp.projects.hoplite import index_jax

index_jax.build_sharded_index(
  db,
  shard_size=500_000,
  shard_degree_bound=128,
  degree_bound=512,
  max_delegates=256,
  alpha=1.5,
  num_steps=-1,
  random_seed=42,
  max_violations=1,
  sample_size=0,
  )
```

Once indexing is complete, we can search like so:

```
from chirp.projects.hoplite import index

hoplite_index = index.HopliteSearchIndex(db)
results, _ = hoplite_index.greedy_search(query, db.get_embedding)
for r in results:
  print(r.embedding_id, r.sort_score)
```
