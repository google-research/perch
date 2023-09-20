This is a micro-library for managing labels from various domains, and
transformations between different sets of labels.

There are a variety of problems which arise in label management. Some of these
problems are intermittent for a given domain, but are exacerbated in areas where
there are many (thousands) of labels. For example:

*   Multiple taxonomies: When multiple taxonomies exist in a given area, we will
    want methods for easy conversion between the taxonomies.
*   Taxonomic updates: When revisions to taxonomies are published, we need to
    update label sets in a consistent way.
*   Non-standard dataset annotations: Often datasets are annotated with ad hoc
    labels, chosen by the researchers. We want to convert these labels to a
    standard taxonomy.
*   Specific label sets: We often want to handle a specific set of labels, such
    as when training classifiers for different subsets of bird species.

By providing a dedicated library for label handling, we hope to isolate solving
these problems from other code.

# Namespaces, Mappings, and ClassLists.

We provide three high-level objects for managing labels:

*   A **namespace** is a fixed (unordered) set of labels. Think of these as a
    *universe* of labels. Some universes are big (e.g., all bird species as they
    appear in the Clements Taxonomy 2021 revision), and some are small (all
    labels appearing in a small dataset).

*   A **mapping** provides a conversion between two namespaces. In its raw form,
    a mapping is a collection of pairs of labels, which is perfectly general
    (see below).

*   A **class list** is an ordered set of labels, from a specific namespace.
    These are useful for specifying subsets, such as classifier targets.

## Taxonomy database

The **taxonomy database** collects all of the namespace, class lists, and
mapping data into a single object for ease of access. You can instantiate the
database with `db = namespace_db.load_db()`. The database itself is cached, so
it is effectively zero-cost to load after the first time it is created in a
program.

## Data storage

The taxonomy data is stored as a plain text database in the form of a large JSON
file. This makes it easy to see the revision history of the database.

## Data consistency

When a taxonomy database is loaded or saved, it is automatically tested for
consistency. This means, e.g., that the labels in a class list are a member of
the namespace that the class list belongs to.

## Recommendations

*   When multiple namespaces are available for a project, we recommend choosing
    a *canonical* namespace, and then converting labels to that target. Creating
    and maintaining conversion tables is a lot of work, and often error-prone.
    Given N different Namespaces, there are N² possible mappings between them.
    But with a canonical target namespace, only N mappings need to be maintained
    (in theory).

*   We also recommend leaving labels on the original data 'at rest.' This
    provides a clear record of the original label, which can then be converted
    for usage. If we later update the conversion (e.g., check in a change in a
    mapping due to a bug) or change our choice of canonical namespace (e.g.,
    update from eBird 2021 to eBird 2022), we can then be sure we haven't lost
    data.

## Details

Note that mappings are assumed to be 1:1, so each label in the source namespace
maps to exactly one label in the target namespace.

Mappings can be composed, although this should be done with care: There is no
guarantee that composed mappings give the same result as direct mappings.
