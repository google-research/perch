# coding=utf-8
# Copyright 2024 The Perch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load eBird/Clements labels from source data."""
import typing

from absl import app
from absl import flags
from chirp.taxonomy import namespace
from chirp.taxonomy import namespace_db
import numpy as np
import pandas as pd

_SOURCE_FILE = flags.DEFINE_string(
    'source_file', 'source_data/ebird_taxonomy_v2022.csv', 'CSV file to load.'
)
_PREFIX = flags.DEFINE_string(
    'prefix',
    'ebird2022',
    'The prefix to attach to the generated namespaces, class lists, and'
    ' mappings.',
)
_OUTPUT_FILE = flags.DEFINE_string(
    'output_file', 'taxonomy_database.json', 'Output file.'
)

SEABIRD_FAMILIES = {
    'sulidae',
    'fregatidae',
    'stercorariidae',
    'laridae',
    'alcidae',
    'scolopacidae',
}

SEABIRD_ORDERS = {
    'sphenisciformes',
    'procellariiformes',
}


def parse_ebird(
    source_file: str | typing.TextIO, prefix: str
) -> namespace_db.TaxonomyDatabase:
  """Parse an eBird CSV source file.

  This parses the CSV file and generates a taxonomy database containing
  namespaces for all eBird codes and for all species codes. It also contains
  namespaces for the scientific names of the species, genera, families, and
  orders. A separate namespace is created containing all the identifiable
  subspecific groups (ISSFs).

  Mappings are created to map eBird codes to their species, genus, family, and
  order.

  Lastly, a class list is created that contains all seabird species.

  Args:
    source_file: The path or file-like object containing the source file.
    prefix: The prefix to use for all the generated data (e.g., `ebird2021` or
      `ebird2022`, to distinguish between versions).

  Returns:
    A `TaxonomyDatabase` containing all the generated data.
  """

  # Load the CSV data
  df = pd.read_csv(source_file)
  # Lower-case the data
  df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
  # Extract the genus from the scientific name
  df['genus'] = df['SCI_NAME'].str.split(' ').str[0]
  # Only keep the scientific family name (ignore the common name)
  df['family'] = df['FAMILY'].str.split(' ').str[0]
  # Correction to spuhs
  df.loc[
      (df['CATEGORY'] == 'spuh')
      & ((df['genus'] == df['ORDER1']) | (df['genus'] == df['family'])),
      'genus',
  ] = np.nan
  # Report species as themselves
  df.loc[df['CATEGORY'] == 'species', 'REPORT_AS'] = df.loc[
      df['CATEGORY'] == 'species', 'SPECIES_CODE'
  ]

  # Namespaces (dictionary key is the name of the namespace)
  namespaces = {
      '': df['SPECIES_CODE'],
      'species': df.loc[df['CATEGORY'] == 'species', 'SPECIES_CODE'],
      'issf': df.loc[df['CATEGORY'] == 'issf', 'SPECIES_CODE'],
      'genera': df['genus'].drop_duplicates().dropna(),
      'families': df['family'].drop_duplicates().dropna(),
      'orders': df['ORDER1'].drop_duplicates().dropna(),
      'clements': df.loc[df['CATEGORY'] == 'species', 'SCI_NAME'],
  }

  # The keys are (mapping name, source namespace, target namespace)
  mappings = {
      ('to_species', '', 'species'): df[
          # Only select rows which should be reported as a species
          df.merge(
              df, left_on='REPORT_AS', right_on='SPECIES_CODE', how='left'
          )['CATEGORY_y']
          == 'species'
      ][['SPECIES_CODE', 'REPORT_AS']],
      ('clements_to_species', 'clements', 'species'): df[
          df['CATEGORY'] == 'species'
      ][['SCI_NAME', 'SPECIES_CODE']],
  }

  for mask, suffix in (
      (df['CATEGORY'] == 'species', 'species'),
      (slice(None), ''),
  ):
    prefix_ = suffix + '_' if suffix else ''
    mappings |= {
        (prefix_ + 'to_genus', suffix, 'genera'): df[mask][
            ['SPECIES_CODE', 'genus']
        ],
        (prefix_ + 'to_family', suffix, 'families'): df[mask][
            ['SPECIES_CODE', 'family']
        ],
        (prefix_ + 'to_order', suffix, 'orders'): df[mask][
            ['SPECIES_CODE', 'ORDER1']
        ],
    }

  if SEABIRD_FAMILIES - set(df['family']):
    raise ValueError('seabird families not found in eBird data')
  if SEABIRD_ORDERS - set(df['ORDER1']):
    raise ValueError('seabird orders not found in eBird data')
  seabirds = df[
      df['family'].isin(SEABIRD_FAMILIES) | df['ORDER1'].isin(SEABIRD_ORDERS)
  ]
  # The keys are class list name, namespace
  class_lists = {
      ('global_seabirds', 'species'): seabirds.loc[
          seabirds['CATEGORY'] == 'species', 'SPECIES_CODE'
      ],
  }

  # Add the prefixes and create the database
  add_prefix = lambda name: (prefix + '_' + name).strip('_')
  namespaces_ = {}
  for name, classes in namespaces.items():
    namespaces_[add_prefix(name)] = namespace.Namespace(frozenset(classes))

  class_lists_ = {}
  for (name, ns), classes in class_lists.items():
    class_lists_[add_prefix(name)] = namespace.ClassList(
        add_prefix(ns), tuple(sorted(classes))
    )

  mappings_ = {}
  for (name, source_ns, target_ns), mapping in mappings.items():
    # Some spuhs don't have a genus, and this was set to nan. Drop these from
    # the mappings.
    mapping = mapping.dropna()
    mappings_[add_prefix(name)] = namespace.Mapping(
        add_prefix(source_ns),
        add_prefix(target_ns),
        dict(zip(mapping.iloc[:, 0], mapping.iloc[:, 1])),
    )

  return namespace_db.TaxonomyDatabase(namespaces_, class_lists_, mappings_)


def main(argv: list[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  ebird_db = parse_ebird(_SOURCE_FILE.value, _PREFIX.value)

  # Merge into existing database and write
  db = namespace_db.load_db()
  db.namespaces |= ebird_db.namespaces
  db.mappings |= ebird_db.mappings
  db.class_lists |= ebird_db.class_lists

  with open(_OUTPUT_FILE.value, 'w') as f:
    f.write(namespace_db.dump_db(db))


if __name__ == '__main__':
  app.run(main)
