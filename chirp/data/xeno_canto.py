# coding=utf-8
# Copyright 2022 The Chirp Authors.
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

"""Utility functions for working with Xeno-Canto species."""

import collections
import concurrent.futures
import dataclasses
import hashlib
import io
import subprocess
import sys
from typing import Dict, FrozenSet, Sequence, Tuple

from absl import logging
from etils import epath
import pandas as pd
import ratelimiter
import requests
import SPARQLWrapper
import tqdm

_EBIRD_TAXONOMY_CHECKSUM = (
    'b007a53bf43e401b6f217b15f78d574669af63dae913c670717bde2c56ea829b')
_EBIRD_TAXONOMY_URL = ('https://www.birds.cornell.edu/clementschecklist/'
                       'wp-content/uploads/2021/08/eBird_Taxonomy_v2021.csv')
_WIKIDATA_CHECKSUM = (
    '9b54e3a35dc6b2fae34ae22f3b16458ea5348d97dd0b2230073679041ba22eb3')
_WIKIDATA_QUERY = """\
SELECT DISTINCT ?item ?itemLabel ?Xeno_canto_species_ID ?eBird_taxon_ID WHERE {
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE]". }
  {
    SELECT DISTINCT ?item WHERE {
      ?item p:P2426 ?statement0.
      ?statement0 ps:P2426 _:anyValueP2426.
      ?item p:P3444 ?statement1.
      ?statement1 ps:P3444 _:anyValueP3444.
    }
  }
  OPTIONAL { ?item wdt:P2426 ?Xeno_canto_species_ID. }
  OPTIONAL { ?item wdt:P3444 ?eBird_taxon_ID. }
}"""
_WIKIDATA_URL = 'https://query.wikidata.org/sparql'
# The Xeno-Canto API allows up to 10 requests per second, but we keep a safety
# margin.
_XC_API_RATE_LIMIT = 8
_XC_API_URL = 'http://www.xeno-canto.org/api/2/recordings'
_XC_SPECIES_URL = 'https://xeno-canto.org/collection/species/all'

# Some of these are resolved through the 2021 eBird taxonomy update webpage:
# https://ebird.org/science/use-ebird-data/2021-ebird-taxonomy-update.
_SCIENTIFIC_NAME_TO_SPECIES_CODES_OVERRIDES = {
    # This script's automations create species code collisions that need
    # resolving for the following species.
    'Anas crecca': 'egwtea1',
    'Anas carolinensis': 'agwtea1',
    'Campylopterus curvipennis': 'wetsab2',
    'Campylopterus pampa': 'wetsab3',
    'Cacicus microrhynchus': 'scrcac2',
    'Cacicus uropygialis': 'scrcac4',
    'Amazilia rutila': 'cinhum1',
    'Calonectris borealis': 'corshe1',
    'Calonectris diomedea': 'scoshe1',
    'Cisticola aberrans': 'rolcis3',
    'Cisticola emini': 'rolcis1',
    'Cyanocorax luxuosus': 'grnjay1',
    'Cyanocorax yncas': 'grnjay2',
    'Cyornis banyumas': 'hibfly3',
    'Phylloscopus maforensis': 'isllew9',
    'Piranga flava': 'heptan3',
    'Piranga hepatica': 'heptan1',
    'Poicephalus robustus': 'brnpar2',
    'Pyrocephalus rubinus': 'verfly1',
    'Pyrocephalus obscurus': 'verfly7',
    # The following species were manually matched to their eBird code.
    'Aegithalos exilis': 'pygtit1',
    'Aethomyias nigrorufus': 'bimwar1',
    'Afrotis afra': 'blabus3',
    'Afrotis afraoides': 'whqbus1',
    'Amazilia wagneri': 'gnfhum2',
    'Caprimulgus centralasicus': 'eurnig1',  # See eBird 2021 taxonomy update.
    'Ceblepyris cinereus': 'ashcus2',
    'Chaetura viridipennis': 'chaswi2',
    'Chlorophoneus bocagei': 'gygbus1',
    'Chlorophoneus sulfureopectus': 'subbus1',
    'Chlorostilbon bracei': 'braeme2',
    'Chlorostilbon elegans': 'braeme3',
    'Cincloramphus mariae': 'necgra1',
    'Cnemotriccus sp.nov.': 'fusfly2',
    'Coloeus monedula': 'eurjac',
    'Corvus caurinus': 'amecro',  # See eBird's 2021 taxonomy update.
    'Cossypha ansorgei': 'anccha1',
    'Cryptomicroeca flaviventris': 'yebrob1',
    'Devioeca papuana': 'canfly2',
    'Edolisoma remotum': 'cicada6',
    'Excalfactoria chinensis': 'blbqua1',
    'Gennaeodryas placens': 'olyrob1',
    'Geothlypis velata': 'masyel5',
    'Glycifohia notabilis': 'nehhon1',
    'Heliobletus sp.nov.Lontras': 'bahtre1',
    'Hemispingus auricularis': 'bkchem2',
    'Hemispingus ochraceus': 'bkehem3',
    'Hemispingus piurae': 'bkehem1',
    'Herpsilochmus sp.nov.Inambari_Tambopata': 'intant1',
    'Himantopus melanurus': 'bknsti2',
    'Hoploxypterus cayanus': 'pielap1',
    'Kempiella griseoceps': 'yelfly4',
    'Lonchura leucosticta': 'sthmun3',
    'Lonchura nigriceps': 'bawman3',
    'Lophotis ruficrista': 'recbus1',
    'Mascarenotus grucheti': 'reusco1',
    'Mascarenotus murivorus': 'rodsco1',
    'Mascarenotus sauzieri': 'mausco1',
    'Melaenornis mariquensis': 'marfly1',
    'Melionyx fuscus': 'soomel1',
    'Melionyx nouhuysi': 'shbmel1',
    'Melionyx princeps': 'lobmel1',
    'Muscicapa itombwensis': 'chafly4',
    'Myiornis sp.nov.Maranhao_Piaui': 'mappyt1',
    'Napothera crispifrons': 'limwrb3',
    'Pampusana jobiensis': 'wbgdov1',
    'Pampusana kubaryi': 'cigdov1',
    'Pampusana rubescens': 'margrd1',
    'Pampusana stairi': 'frgdov1',
    'Phacellodomus tax.nov.': 'mantho1',
    'Picumnus fulvescens': 'ochpic1',
    'Picumnus nigropunctatus': 'gospic3',
    'Porzana atra': 'heicra1',
    'Porzana nigra': 'milrai1',
    'Pseudobulweria rupinarum': 'lshpet1',
    'Psittacula bensoni': 'magpar1',
    'Ptyrticus turdinus': 'thrbab1',
    'Rallicula mayri': 'mayrai1',
    'Ramphastos citreolaemus': 'chbtou3',
    'Scytalopus sp.nov.Ampay': 'amptap1',
    'Scytalopus sp.nov.Millpo': 'miltap1',
    'Setophaga auduboni': 'audwar',
    'Silvicultrix spodionota': 'crocht3',
    'Thalasseus acuflavidus': 'santer2',
    'Thamnolaea coronata': 'moccha1',
    'Thapsinillas longirostris': 'sulgob1',
    'Trichoglossus flavoviridis': 'yaglor2',
    'Trochilus scitulus': 'stream3',
    'Zosterops tax.nov.Wangi.wangi': 'wawwhe1',
}
_ALLOWED_SPECIES_CODE_COLLISIONS = {
    # See eBird's 2021 taxonomy update regarding Vaurie's Nightjar.
    frozenset(('Caprimulgus europaeus', 'Caprimulgus centralasicus')):
        'eurnig1',
    # See eBird's 2021 taxonomy update regarding Northwestern Crow.
    frozenset(('Corvus brachyrhynchos', 'Corvus caurinus')):
        'amecro',
    # Both are considered subspecies within Chaeture chapmani in eBird's 2021
    # taxonomy update.
    frozenset(('Chaetura chapmani', 'Chaetura viridipennis')):
        'chaswi2',
    # Picumnus fulvescens is no longer recognized as a subspecies in eBird's
    # 2021 taxonomy update.
    frozenset(('Picumnus fulvescens', 'Picumnus limae')):
        'ochpic1',
    # According to Wikipedia (which cites Birds of the World), Thamnolaea
    # coronata is sometimes considered a subspecies of Thamnolaea
    # cinnamomeiventris.
    frozenset(('Thamnolaea cinnamomeiventris', 'Thamnolaea coronata')):
        'moccha1',
}


@dataclasses.dataclass
class SpeciesMappingConfig:
  """Configuration values for mapping Xeno-Canto species to eBird codes."""
  # Manual species code assignments.
  scientific_name_to_species_code_overrides: Dict[str, str] = dataclasses.field(
      default_factory=lambda: _SCIENTIFIC_NAME_TO_SPECIES_CODES_OVERRIDES)
  # Species for which a species code is not required (usually because they are
  # extinct).
  species_with_no_code: Sequence[str] = ('Cyanoramphus erythrotis',
                                         'Cyanoramphus subflavescens',
                                         'Nycticorax olsoni')
  # Sets of species for which a species code collision is allowed (usually
  # because they are merged into one species in eBird's taxonomy).
  allowed_species_code_collisions: Dict[
      FrozenSet[str], str] = dataclasses.field(
          default_factory=lambda: _ALLOWED_SPECIES_CODE_COLLISIONS)
  # Insect genera which should be ignored.
  insect_genera: Sequence[str] = (
      'acheta', 'amphiestris', 'anonconotus', 'chorthippus', 'chrysochraon',
      'conocephalus', 'decorana', 'decticus', 'euchorthippus', 'eupholidoptera',
      'gomphocerippus', 'gryllodes', 'gryllodinus', 'gryllotalpa', 'gryllus',
      'incertana', 'isophya', 'leptophyes', 'meconema', 'metrioptera',
      'myrmeleotettix', 'nemobius', 'ochrilidia', 'oecanthus', 'omocestus',
      'parnassiana', 'phaneroptera', 'pholidoptera', 'platycleis', 'poecilimon',
      'pseudochorthippus', 'pterolepis', 'rhacocleis', 'roeseliana', 'ruspolia',
      'sardoplatycleis', 'sporadiana', 'stenobothrus', 'stethophyma',
      'tettigonia', 'zeuneriana')


@dataclasses.dataclass
class RecordingInfo:
  # Xeno-Canto recording ID.
  xc_id: str
  # Quality score in {'A', 'B', 'C', 'D', 'E', '', 'no score'}.
  quality_score: str
  # Background species (scientific names).
  background_species: Sequence[str]


def _infer_species_codes_from_wikidata(taxonomy_info: pd.DataFrame,
                                       overrides: Dict[str, str]) -> pd.Series:
  """Infers each species' code from data obtained through a Wikidata query.

  Args:
    taxonomy_info: Xeno-Canto taxonomy DataFrame.
    overrides: dict used to override species code mappings obtained from
      Wikidata.

  Returns:
    The species codes for each row in the species table, with NaN values when
    the code cannot be found via Wikidata.

  Raises:
    RuntimeError: if the Wikidata query's return value has an unexpected
      checksum.
  """
  sparql = SPARQLWrapper.SPARQLWrapper(
      _WIKIDATA_URL,
      agent=('ChirpBot/0.1 (https://github.com/google-research/chirp) '
             f'Python/{sys.version_info.major}/{sys.version_info.minor}'))
  sparql.setQuery(_WIKIDATA_QUERY)
  sparql.setReturnFormat(SPARQLWrapper.JSON)
  matches = sorted(
      (item['eBird_taxon_ID']['value'], item['Xeno_canto_species_ID']['value'])
      for item in sparql.queryAndConvert()['results']['bindings'])

  if hashlib.sha256(
      str(matches).encode('utf-8')).hexdigest() != _WIKIDATA_CHECKSUM:
    raise RuntimeError('Return value for the Wikidata SPARQL query has the '
                       'wrong checksum value.')

  scientific_name_to_species_code = {
      name.replace('-', ' '): code for code, name in matches
  }
  scientific_name_to_species_code.update(overrides)

  return taxonomy_info['Scientific name'].map(scientific_name_to_species_code)


def _load_ebird_taxonomy() -> pd.DataFrame:
  """Returns the eBird taxonomy as a DataFrame."""
  # requests.get(...) tends to cut the response short, leading to an incomplete
  # CSV file.
  ebird_taxonomy = pd.read_csv(
      io.StringIO(
          subprocess.run(['curl', _EBIRD_TAXONOMY_URL],
                         check=True,
                         stdout=subprocess.PIPE).stdout.decode('utf-8')))
  if hashlib.sha256(ebird_taxonomy.to_json().encode(
      'utf-8')).hexdigest() != _EBIRD_TAXONOMY_CHECKSUM:
    raise RuntimeError('eBird taxonomy has the wrong checksum value.')
  # Sanity check.
  for column in ('SPECIES_CODE', 'SCI_NAME', 'PRIMARY_COM_NAME'):
    if len(ebird_taxonomy[column].unique()) != len(ebird_taxonomy[column]):
      raise RuntimeError(
          f"Expected eBird taxonomy's {column} values to be unique")

  return ebird_taxonomy


def _clean_up_species_codes(taxonomy_info: pd.DataFrame,
                            ebird_taxonomy: pd.DataFrame) -> pd.DataFrame:
  """Cleans up species codes.

  This function

  1) corrects some false negative species code matches by cross-referencing the
     eBird taxonomy for common and scientific names,
  2) marks the fixed species codes for manual verification via the 'to_verify'
     column.

  Args:
    taxonomy_info: Xeno-Canto taxonomy DataFrame.
    ebird_taxonomy: eBird taxonomy DataFrame.

  Returns:
    The cleaned up Xeno-Canto taxonomy DataFrame.
  """
  # To reduce false negative matches, we normalize common and scientific names
  # to lowercase, and we remove spaces and dashes from common names.
  format_common_name = lambda s: s.lower().replace('-', '').replace(' ', '')
  format_scientific_name = lambda s: s.lower()
  common_names = map(format_common_name, ebird_taxonomy['PRIMARY_COM_NAME'])
  common_name_to_code = dict(zip(common_names, ebird_taxonomy['SPECIES_CODE']))
  scientific_names = map(format_scientific_name, ebird_taxonomy['SCI_NAME'])
  scientific_name_to_code = dict(
      zip(scientific_names, ebird_taxonomy['SPECIES_CODE']))

  def _address_edge_cases(row):
    # We try to salvage by matching scientific names or common names.
    if row.isna()['species_code'] and not row['no_species_code']:
      scientific_name = format_scientific_name(row['Scientific name'])
      scientific_words = scientific_name.split(' ')
      common_name = format_common_name(row['Common name'])
      alternative_common_names = (
          # Sometimes a difference in UK vs US spelling causes a false negative
          # match.
          common_name.replace('grey', 'gray'),
          # Some species are of an 'undescribed form' on eBird.
          common_name + '(undescribedform)')

      if scientific_name in scientific_name_to_code:
        row['species_code'] = scientific_name_to_code[scientific_name]
      elif common_name in common_name_to_code:
        row['species_code'] = common_name_to_code[common_name]
      elif any(n in common_name_to_code for n in alternative_common_names):
        common_name = next(
            n for n in alternative_common_names if n in common_name_to_code)
        row['species_code'] = common_name_to_code[common_name]
      else:
        matches = [
            k for k in scientific_name_to_code
            if all(w in k for w in scientific_words)
        ]
        if len(matches) == 1:
          scientific_name, = matches
          species_code = scientific_name_to_code[scientific_name]
          row['species_code'] = species_code

      # If we managed to find a match, flag it for manual verification.
      row['to_verify'] = not row.isna()['species_code']

    return row

  return taxonomy_info.apply(_address_edge_cases, axis=1)


def _ensure_species_codes_uniqueness(
    taxonomy_info: pd.DataFrame, allowed_collisions: Dict[FrozenSet[str],
                                                          str]) -> pd.DataFrame:
  """Ensures the species codes are unique.

  If a collision is allowed for a specific code, the colliding rows are merged
  and assigned to the first colliding row, and the other colliding rows are
  dropped.

  Args:
    taxonomy_info: Xeno-Canto taxonomy DataFrame.
    allowed_collisions: mapping from sets of scientific names to their
      corresponding species code indicating which collisions are allowed.

  Returns:
    Xeno-Canto taxonomy DataFrame, with the allowed collisions addressed.

  Raises:
    ValueError: if there are disallowed species code collisions.
  """
  taxonomy_info = taxonomy_info.copy()
  species_code_counter = collections.Counter(taxonomy_info['species_code'])
  collisions = [k for k, v in species_code_counter.items() if v > 1]
  if collisions:
    for species_code in collisions:
      colliding_rows = taxonomy_info[taxonomy_info['species_code'] ==
                                     species_code]
      scientific_names = frozenset(colliding_rows['Scientific name'])
      if (scientific_names in allowed_collisions and
          allowed_collisions[scientific_names] == species_code):
        # We merge colliding rows into a single row.
        merged_row = pd.Series({
            'Common name': ','.join(colliding_rows['Common name']),
            'Scientific name': ','.join(colliding_rows['Scientific name']),
            'No.': colliding_rows['No.'].sum(),
            'No. Back': colliding_rows['No. Back'].sum(),
            'species_code': species_code,
            'is_insect': colliding_rows['is_insect'].any(),
            'no_species_code': colliding_rows['no_species_code'].any(),
            'to_verify': colliding_rows['to_verify'].any(),
        })
        taxonomy_info.loc[colliding_rows.index[0]] = merged_row
        taxonomy_info = taxonomy_info.drop(index=colliding_rows.index[1:])
      else:
        raise ValueError('Species code are not unique.')
  return taxonomy_info


def _break_down_taxonomy(taxonomy_info: pd.DataFrame,
                         ebird_taxonomy: pd.DataFrame) -> pd.DataFrame:
  """Breaks down the species taxonomy into multiple columns.

  Args:
    taxonomy_info: Xeno-Canto taxonomy DataFrame.
    ebird_taxonomy: eBird taxonomy DataFrame.

  Returns:
    The Xeno-Canto species, with [...].
  """
  taxonomy_info = taxonomy_info.copy()
  taxonomy_info['xeno_canto_query'] = taxonomy_info['Scientific name']
  # Use eBird's scientific names for the taxonomical breakdown.
  taxonomy_info['scientific_name'] = taxonomy_info['species_code'].map(
      dict(zip(ebird_taxonomy['SPECIES_CODE'], ebird_taxonomy['SCI_NAME'])))
  # The species is the second word in the scientific name, but sometimes
  # scientific names are of the form '<GENUS> [undescribed form]', which is
  # why we use all but the first word.
  taxonomy_info['species'] = taxonomy_info['scientific_name'].map(
      lambda s: ' '.join(s.lower().split(' ')[1:]))
  # The genus is the first word in the scientific name.
  taxonomy_info['genus'] = taxonomy_info['scientific_name'].map(
      lambda s: s.lower().split(' ')[0])
  # The family name is of the form '<FAMILY> (<MORE COMMON NAME>)', which is why
  # we only keep the first word.
  taxonomy_info['family'] = taxonomy_info['species_code'].map(
      dict(
          zip(ebird_taxonomy['SPECIES_CODE'],
              ebird_taxonomy['FAMILY']))).map(lambda s: s.lower().split(' ')[0])
  taxonomy_info['order'] = taxonomy_info['species_code'].map(
      dict(zip(ebird_taxonomy['SPECIES_CODE'],
               ebird_taxonomy['ORDER1']))).map(lambda s: s.lower())
  taxonomy_info['common_name'] = taxonomy_info['species_code'].map(
      dict(
          zip(ebird_taxonomy['SPECIES_CODE'],
              ebird_taxonomy['PRIMARY_COM_NAME']))).map(lambda s: s.lower())
  return taxonomy_info


def _scrape_xeno_canto_recording_metadata(
    taxonomy_info: pd.DataFrame,
    progress_bar: bool = True) -> Dict[str, Sequence[RecordingInfo]]:
  """Scrapes and returns Xeno-Canto recording metadata for all species.

  Args:
    taxonomy_info: Xeno-Canto taxonomy DataFrame.
    progress_bar: whether to show a progress bar.

  Returns:
    A mapping from species code to Xeno-Canto recording metadata for that
    species.
  """
  session = requests.Session()
  session.mount(
      'http://',
      requests.adapters.HTTPAdapter(
          max_retries=requests.adapters.Retry(total=5, backoff_factor=0.1)))

  @ratelimiter.RateLimiter(max_calls=_XC_API_RATE_LIMIT, period=1)
  def retrieve_recording_info_dicts(row):
    species_code = row['species_code']
    recording_info_dicts = []
    for query in row['xeno_canto_query'].replace(' ', '%20').split(','):
      response = session.get(
          # Specifying gen:<GENUS> speeds up the lookup.
          url=f"{_XC_API_URL}?query={query}%20gen:{row['genus']}").json()
      dicts = response['recordings']
      # If there are more than one page of responses, loop over pages.
      if int(response['numPages']) > 1:
        for page in range(2, int(response['numPages']) + 1):
          # Specifying gen:<GENUS> speeds up the lookup.
          url_with_page = (
              f"{_XC_API_URL}?query={query}%20gen:{row['genus']}&page={page}")
          dicts.extend(session.get(url=url_with_page).json()['recordings'])
      if len(dicts) != int(response['numRecordings']):
        raise RuntimeError(
            f'wrong number of recordings obtained for {species_code}')
      recording_info_dicts.extend(dicts)
    return (species_code, recording_info_dicts)

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=_XC_API_RATE_LIMIT) as executor:
    rows = [row for _, row in taxonomy_info.iterrows()]
    iterator = executor.map(retrieve_recording_info_dicts, rows)
    if progress_bar:
      iterator = tqdm.tqdm(iterator, total=len(rows))
    species_code_to_recording_info_dicts = dict(iterator)

  species_code_to_recording_metadata = {}
  num_total, num_restricted, num_nd, num_remaining = 0, 0, 0, 0
  for species_code, dicts in species_code_to_recording_info_dicts.items():
    num_total += len(dicts)
    # Avoid restricted recordings and *-nd licenses. Restricted recordings
    # have an empty string as the 'file' value.
    num_restricted += len([r for r in dicts if not r['file']])
    num_nd += len([r for r in dicts if '-nd' in r['lic']])
    recording_info_dicts = [
        RecordingInfo(r['id'], r['q'], r['also'])
        for r in dicts
        if r['file'] and '-nd' not in r['lic']
    ]

    num_remaining += len(recording_info_dicts)
    species_code_to_recording_metadata[species_code] = recording_info_dicts

  logging.info(
      'Retrieved %d recordings out of %d, ignoring %d restricted recordings '
      'and %d ND-licensed recordings', num_remaining, num_total, num_restricted,
      num_nd)

  return species_code_to_recording_metadata


def create_taxonomy_info(
    species_mapping_config: SpeciesMappingConfig) -> pd.DataFrame:
  """Creates a taxonomy DataFrame for Xeno-Canto species.

  The DataFrame is populated with the following columns:

  * 'species_code': eBird species code.
  * 'xeno_canto_query': query by which to search for recordings on Xeno-Canto.
  * 'scientific_name': scientific name.
  * 'species': species name.
  * 'genus': genus name.
  * 'family': family name.
  * 'order': order name.
  * 'common_name': common name.

  Args:
    species_mapping_config: configuration values used to create the mapping from
      species name to species code.

  Returns:
    A taxonomy DataFrame for Xeno-Canto species.
  """
  # Download Xeno-Canto species table.
  # Columns: 'Common name', 'Scientific name', 'Status', 'No.', 'No. Back'.
  logging.info('Downloading Xeno-Canto species list...')
  taxonomy_info, = pd.read_html(io=_XC_SPECIES_URL, match='Scientific name')

  # We drop this unused column early because it contains NaNs and we want to use
  # DataFrame.dropna() later on.
  # Columns: 'Common name', 'Scientific name', 'No.', 'No. Back'.
  taxonomy_info = taxonomy_info.drop(columns='Status')

  # We first retrieve all Wikidata items with both an `eBird taxon ID` and a
  # `Xeno-canto species ID`. This maps ~ 93% of Xeno-Canto species names to
  # their corresponding eBird species code.
  # Columns: 'Common name', 'Scientific name', 'No.', 'No. Back',
  #          'species_code'.
  logging.info('Querying Wikidata to infer species codes...')
  overrides = species_mapping_config.scientific_name_to_species_code_overrides
  taxonomy_info['species_code'] = _infer_species_codes_from_wikidata(
      taxonomy_info, overrides=overrides)

  # Sometimes Wikidata causes false positives by matching a Xeno-Canto species
  # to a nonexistent eBird code. To remove false positives, we create an
  # identity mapping for all eBird taxonomy codes; keys not in the dict are
  # mapped to NaN.
  logging.info('Downloading eBird taxonomy...')
  ebird_taxonomy = _load_ebird_taxonomy()
  taxonomy_info['species_code'] = taxonomy_info['species_code'].map(
      dict(zip(ebird_taxonomy['SPECIES_CODE'], ebird_taxonomy['SPECIES_CODE'])))

  # Yes, Xeno-Canto has some insect songs, too! We ignore those. We also ignore
  # extinct species with no eBird code listed in
  # `species_mapping_config.species_with_no_code`.
  # Columns: 'Common name', 'Scientific name', 'No.', 'No. Back',
  #          'species_code', 'is_insect', 'no_species_code'.
  insect_genera = species_mapping_config.insect_genera
  taxonomy_info['is_insect'] = taxonomy_info['Scientific name'].map(
      lambda s: any(genus in s.lower() for genus in insect_genera))
  taxonomy_info['no_species_code'] = (
      taxonomy_info['is_insect']
      | taxonomy_info['Scientific name'].isin(
          species_mapping_config.species_with_no_code))

  # Flag overridden species codes for manual verification.
  # Columns: 'Common name', 'Scientific name', 'No.', 'No. Back',
  #          'species_code', 'is_insect', 'no_species_code', 'to_verify'.
  taxonomy_info['to_verify'] = taxonomy_info['Scientific name'].isin(overrides)

  # We clean up false negatives by matching common and scientific names between
  # Xeno-Canto and eBird's taxonomy.
  logging.info('Cleaning up species codes...')
  taxonomy_info = _clean_up_species_codes(taxonomy_info, ebird_taxonomy)

  # Drop insect-related species and bird species whose eBird code cannot be
  # found.
  insect_species = taxonomy_info[
      taxonomy_info['species_code'].isna()
      & taxonomy_info['is_insect']]['Common name'].tolist()
  logging.info('Ignoring the following %d insect species:\n- %s',
               len(insect_species), '\n- '.join(insect_species))

  to_verify = taxonomy_info[taxonomy_info['to_verify']]
  to_verify_xeno_canto = to_verify['Common name']
  to_verify_ebird = to_verify['species_code'].map(
      dict(
          zip(ebird_taxonomy['SPECIES_CODE'],
              ebird_taxonomy['PRIMARY_COM_NAME'])))

  # `allowlist` is a set of (species code, scientific name) tuples representing
  # species code matches which have already been manually validated. The
  # rationale for each match is detailed in `species_code_rationales.json`.
  file_path = (
      epath.Path(__file__).parent / 'species_code_rationales.json')
  allowlist = pd.read_json(file_path, orient='index')
  allowlist = set(
      zip(allowlist['species_code'], allowlist['xc_scientific_name']))

  needs_manual_validation = []
  for k, v, s, c in zip(to_verify_xeno_canto, to_verify_ebird,
                        to_verify['Scientific name'],
                        to_verify['species_code']):
    # When common names match exactly, we don't need manual validation.
    if k != v and (c, s) not in allowlist:
      needs_manual_validation.append((k, v))

  if needs_manual_validation:
    logging.info(
        'The following %d species need manual verification (keys are '
        'from Xeno-Canto, values are from eBird):',
        len(needs_manual_validation))
    for k, v in needs_manual_validation:
      logging.info('- %s:\n      %s', k, v)

  not_found = taxonomy_info[
      taxonomy_info['species_code'].isna()
      & ~taxonomy_info['no_species_code']]['Scientific name'].tolist()
  if not_found:
    logging.warning(
        'Ignoring the following %d bird species because their eBird code '
        'cannot be found: %s', len(not_found), not_found)
  taxonomy_info = taxonomy_info.dropna()

  # Make sure the species codes are unique. If a collision is allowed for a
  # specific code, merge the colliding rows.
  taxonomy_info = _ensure_species_codes_uniqueness(
      taxonomy_info,
      allowed_collisions=species_mapping_config.allowed_species_code_collisions)

  # Map species code to species, genus, family, order, and common name.
  # Columns: 'Common name', 'Scientific name', 'No.', 'No. Back',
  #          'species_code', 'is_insect', 'no_species_code', 'to_verify',
  #          'xeno_canto_query', 'scientific_name', 'species', 'genus',
  #          'family', 'order', 'common_name'.
  taxonomy_info = _break_down_taxonomy(taxonomy_info, ebird_taxonomy)

  # Drop unused columns.
  # Columns: 'species_code', 'xeno_canto_query', 'scientific_name', 'species',
  #          'genus', 'family', 'order', 'common_name'.
  taxonomy_info = taxonomy_info.drop(columns=[
      'Common name', 'Scientific name', 'No.', 'No. Back', 'is_insect',
      'no_species_code', 'to_verify'
  ])

  return taxonomy_info


def retrieve_recording_metadata(taxonomy_info: pd.DataFrame,
                                progress_bar: bool = True) -> pd.DataFrame:
  """Retrieves recording metadata for a given Xeno-Canto taxonomy DataFrame.

  The input DataFrame expected to be populated with the following columns:

  * 'species_code': eBird species code.
  * 'xeno_canto_query': query by which to search for recordings on Xeno-Canto.
  * 'scientific_name': scientific name.
  * 'species': species name.
  * 'genus': genus name.
  * 'family': family name.
  * 'order': order name.
  * 'common_name': common name.

  This function adds the following columns:

  * 'xeno_canto_ids': list of Xeno-Canto recording IDs.
  * 'xeno_canto_quality_scores': list of recording quality scores.
  * 'xeno_canto_bg_species_codes': list of species names for species heard in
      the background.

  Args:
    taxonomy_info: Xeno-Canto taxonomy DataFrame.
    progress_bar: whether to show a progress bar.

  Returns:
    A Xeno-Canto taxonomy DataFrame with added recording metadata.
  """
  taxonomy_info = taxonomy_info.copy()
  logging.info('Scraping Xeno-Canto for recording IDs...')
  species_code_to_recording_metadata = _scrape_xeno_canto_recording_metadata(
      taxonomy_info, progress_bar=progress_bar)
  taxonomy_info['xeno_canto_ids'] = taxonomy_info['species_code'].map({
      species_code: [info.xc_id for info in metadata]
      for species_code, metadata in species_code_to_recording_metadata.items()
  })
  taxonomy_info['xeno_canto_quality_scores'] = taxonomy_info[
      'species_code'].map({
          species_code: [info.quality_score for info in metadata] for
          species_code, metadata in species_code_to_recording_metadata.items()
      })

  xeno_canto_query_to_species_code = dict(
      zip(taxonomy_info['xeno_canto_query'], taxonomy_info['species_code']))

  def _to_species_codes(background_species):
    return [
        xeno_canto_query_to_species_code[q]
        for q in background_species
        if q in xeno_canto_query_to_species_code
    ]

  taxonomy_info['xeno_canto_bg_species_codes'] = taxonomy_info[
      'species_code'].map({
          code:
          [_to_species_codes(info.background_species) for info in metadata]
          for code, metadata in species_code_to_recording_metadata.items()
      })

  return taxonomy_info
