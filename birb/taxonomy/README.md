This is a library for managing labels from various domains, and transformations
between different sets of labels.

There are a variety of problems which arise in label management. Some of these
problems are intermittent for a given domain, but are exacerbated in areas
where there are many (thousands) of labels. For example:

* Multiple Taxonomies: When multiple taxonomies exist in a given area, we will
want methods for easy conversion between the taxonomies.
* Taxonomic Updates: When revisions to taxonomies are published, we need to
update label sets in a consistent way.
* Non-Standard Dataset Annotations: Often datasets are annotated with ad hoc
labels, chosen by the researchers. We want to convert these labels to a
standard taxonomy.
* Specific Label Sets: We often want to handle a specific set of labels, such
as when training classifiers for different subsets of bird species.

By providing a dedicated library for label handling, we hope to isolate solving
these problems from other code.

# Namespaces, Mappings, and ClassLists.

We provide three high-level objects for managing labels:

* A **Namespace** is a fixed (unordered) set of labels. Think of these as a
/universe/ of labels. Some universes are big (all bird species, as they appear
in the Clements Taxonomy 2021 revision), and some are small (all labels
appearing in annotations of the Caples soundscape dataset).

* A **Mapping** provides a conversion between two namespaces. In its raw form,
a mapping is a collection of pairs of labels, which is perfectly
general (see below).

* A **ClassList** is an ordered set of labels, from a specific namespace. These
are useful for specifying subsets, such as classifier targets.

## The Namespace DB

The **Namespace DB** collects all of the Namespace, ClassList and Mapping data
into a single object for ease of access. You can instantiate the DB with
`db = namespace_db.load_db()`. The DB itself is cached, so it is effectively
zero-cost to get the DB after the first time it is created in a program.

## Handling Data

These three basic objects are simple to write down and store in
machine-readable forms (such as CSVs).

* If we have a machine-readable source-of-truth (like the eBird taxonomy CSV
file), we can create many namespaces and mappings automatically. We call these
**generators**. Because we work in a modern version-control system, these
generators can be easily updated when we find bugs.

* For hand-curated data, we store data in CSVs, under the `data` directory.
This includes (for example) conversions between labels specific to a particular
dataset and so on. The CSVs have comment fields which allow documenting why
a particular decision was made.

## Data Consistency

We provide tests which automatically checks for data consistency.

For example, all labels in a given Mapping must appear in the respective
source and target namespaces. In case of a test failure, the test logs will
indicate which specific object had problems, and provide a list of offending
labels.

## Recommendations

* When multiple namespaces are available for a project, we recommend choosing a
*canonical* Namespace, and then converting labels to that target. Creating and
maintaining conversion tables is a lot of work, and often error-prone. Given N
different Namespaces, there are `N^2` possible mappings between them. But with a
canonical target namespace, only N mappings need to be maintained (in theory).

* We also recommend leaving labels on the original data 'at rest.' This provides
a clear record of the original label, which can then be converted for usage.
If we later update the conversion (eg, check in a change in a Mapping due to
a bug) or change our choice of canonical namespace (eg, update from ebird2021
to ebird2022), we can then be sure we haven't lost data.

## Advanced Topics in Label Juggling

Most of the complexity is in handling Mappings. For the following, consider a
mapping `M: X -> Y`, where `X` is the **source** namespace and `Y` is the
**target** namespace.

At the end of the day, a Mapping is just an association of labels in two
different sets (the source and target namespaces). There are `|X|*|Y|` possible
pairs of labels in the two namespaces: we only need to track whether each pair
of labels is associated or not. Thus, every Mapping can be encoded as a binary
matrix.

Mappings are not necessarily *functions*. We can have a single source label
associated with multiple target labels (such as happens with a **split** in
a bird taxonomy update). The Mapping can encode that a split has occurred,
but it is up to the downstream user to decide what to do about it (for example,
using additional Lat/Long data or classifier outputs to associate a vocalization
with one of the split species).

* Note that Mappings are defined on **namespaces** not on **ClassLists**.
Given a mapping between namespaces, the same mapping automatically defines
a mapping between ClassLists. Meanwhile, given two ClassLists in the same
namespace, we automatically have a mapping between the two given by a (partial)
permutation, so there's no need to track these explicitly.

* We often find that after applying a mapping it is helpful to keep track
of which labels in the target namespace actually correspond to labels in the
source namespace. This is the set `M(X)`, which we call the **image** of `M`.
As a result, most methods which apply mappings also provide the image set.

* Mappings can be **composed**. Given another mapping `N: Y -> Z`, we can
compose `N` with `M`. The composition has its own image set, `N(M(X))`.

* We often want to realize a Mapping in a particular form, for example as a
Python dictionary (`M.to_dict()`), as a binary matrix, or as a Tensorflow lookup
table. Be aware that some realizations are not general; for example,
dictionaries are one-to-one mappings, so do not handle *splits* well.
