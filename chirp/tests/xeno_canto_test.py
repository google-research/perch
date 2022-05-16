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

"""Tests for xeno_canto."""

import functools
import hashlib
import io
from unittest import mock

from chirp.data import xeno_canto
import pandas as pd

from absl.testing import absltest
from absl.testing import parameterized


def _make_mock_requests_side_effect(broken_down_taxonomy_info,
                                    wrong_num_recordings=False):
  to_query = lambda row: row['xeno_canto_query'].replace(' ', '%20')
  to_genus = lambda row: row['xeno_canto_query'].split(' ')[0]
  api_url = xeno_canto._XC_API_URL
  to_url = lambda row: f'{api_url}?query={to_query(row)}%20gen:{to_genus(row)}'

  xeno_canto_queries = broken_down_taxonomy_info['xeno_canto_query'].tolist()

  def make_recordings(i):
    return [{
        'file': f'XC{i:05d}.mp3',
        'id': f'{i:05d}',
        'q': {
            0: 'A',
            1: '',
            2: 'no score'
        }[i],
        'also': [xeno_canto_queries[(i + 1) % len(broken_down_taxonomy_info)]],
        'lic': 'cc-by'
    }, {
        'file': '',
        'id': f'{i:05d}',
        'q': {
            0: 'A',
            1: '',
            2: 'no score'
        }[i],
        'also': [xeno_canto_queries[(i + 1) % len(broken_down_taxonomy_info)]],
        'lic': 'cc-by'
    }, {
        'file': f'XC{i:05d}.mp3',
        'id': f'{i:05d}',
        'q': {
            0: 'A',
            1: '',
            2: 'no score'
        }[i],
        'also': [xeno_canto_queries[(i + 1) % len(broken_down_taxonomy_info)]],
        'lic': 'cc-by-nd'
    }]

  # Return recordings in two pages to test resiliency to multi-page results.
  return_values = {
      to_url(row): {
          'recordings': make_recordings(i)[:-1],
          'numRecordings': '4' if wrong_num_recordings else '3',
          'numPages': '2',
      } for i, row in broken_down_taxonomy_info.iterrows()
  }
  return_values.update({
      to_url(row) + '&page=2': {
          'recordings': make_recordings(i)[-1:],
          'numRecordings': '4' if wrong_num_recordings else '3',
          'numPages': '2',
      } for i, row in broken_down_taxonomy_info.iterrows()
  })

  def side_effect(url):
    mock_response = mock.MagicMock()
    mock_response.json.return_value = return_values[url]
    return mock_response

  return side_effect


class XenoCantoTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.ebird_taxonomy_csv_str = (
        'TAXON_ORDER,CATEGORY,SPECIES_CODE,PRIMARY_COM_NAME,SCI_NAME,ORDER1,'
        'FAMILY,SPECIES_GROUP,REPORT_AS\n'
        '1,species,ostric2,Common Ostrich,Struthio camelus,Struthioniformes,'
        'Struthionidae (Ostriches),Ostriches,\n'
        '6,species,ostric3,Somali Ostrich,Struthio molybdophanes,'
        'Struthioniformes,Struthionidae (Ostriches),,\n'
        '8,species,grerhe1,Greater Rhea,Rhea americana,Rheiformes,'
        'Rheidae (Rheas),Rheas,')
    self.taxonomy_info = pd.DataFrame({
        'species_code': ['ostric2', 'ostric3', 'grerhe1'],
        'Scientific name': [
            'Struthio camelus', 'Struthio molybdophanes', 'Rhea americana'
        ],
        'Common name': ['Common Ostrich', 'Somali Ostrich', 'Greater Rhea'],
        'No.': [2, 2, 2],
        'No. Back': [1, 1, 1],
        'is_insect': [False, False, False],
        'no_species_code': [False, False, False],
        'to_verify': [False, False, False],
    })

    taxonomy_info = self.taxonomy_info.copy()
    taxonomy_info['xeno_canto_query'] = taxonomy_info['Scientific name']
    taxonomy_info['scientific_name'] = taxonomy_info['Scientific name']
    taxonomy_info['species'] = ['camelus', 'molybdophanes', 'americana']
    taxonomy_info['genus'] = ['struthio', 'struthio', 'rhea']
    taxonomy_info['family'] = ['struthionidae', 'struthionidae', 'rheidae']
    taxonomy_info['order'] = [
        'struthioniformes', 'struthioniformes', 'rheiformes'
    ]
    taxonomy_info['common_name'] = [
        'common ostrich', 'somali ostrich', 'greater rhea'
    ]
    self.broken_down_taxonomy_info = taxonomy_info

  @mock.patch.object(xeno_canto.SPARQLWrapper, 'SPARQLWrapper', autospec=True)
  def test_wikidata_species_codes(self, mock_sparql_wrapper):
    # Simulate a code collision to test the overrides kwarg.
    name_to_code = dict(
        zip(
            self.taxonomy_info['Scientific name'].map(
                lambda n: n.replace(' ', '-')),
            self.taxonomy_info['species_code']))
    first_key, second_key = list(name_to_code.keys())[:2]
    overrides = {second_key.replace('-', ' '): name_to_code[second_key]}
    name_to_code[second_key] = name_to_code[first_key]
    matches = sorted((c, n) for n, c in name_to_code.items())

    # Mock the SPARQL wrapper and have it return a fake response.

    bindings = []
    for n, c in name_to_code.items():
      bindings.append({
          'eBird_taxon_ID': {
              'value': c
          },
          'Xeno_canto_species_ID': {
              'value': n
          }
      })
    mock_sparql_wrapper.return_value.queryAndConvert.return_value = {
        'results': {
            'bindings': bindings
        }
    }

    function_call = functools.partial(
        xeno_canto._infer_species_codes_from_wikidata,
        # Pass in a species table without species codes; we want to infer them.
        taxonomy_info=self.taxonomy_info.drop(columns='species_code'),
        overrides=overrides)

    # We should recover the original species codes.
    checksum = hashlib.sha256(str(matches).encode('utf-8')).hexdigest()
    with mock.patch.object(xeno_canto, '_WIKIDATA_CHECKSUM', new=checksum):
      self.assertListEqual(function_call().to_list(),
                           self.taxonomy_info['species_code'].to_list())

    # Without patching with the correct checksum, the function should raise a
    # RuntimeError.
    self.assertRaises(RuntimeError, function_call)

  @mock.patch.object(xeno_canto.subprocess, 'run', autospec=True)
  def test_load_ebird_taxonomy(self, mock_run):
    mock_run.return_value = mock.MagicMock()

    # The function should return the correct taxonomy.
    mock_run.return_value.stdout = self.ebird_taxonomy_csv_str.encode('utf-8')
    taxonomy = pd.read_csv(io.StringIO(self.ebird_taxonomy_csv_str))
    checksum = hashlib.sha256(taxonomy.to_json().encode('utf-8')).hexdigest()
    with mock.patch.object(
        xeno_canto, '_EBIRD_TAXONOMY_CHECKSUM', new=checksum):
      self.assertTrue(xeno_canto._load_ebird_taxonomy().equals(taxonomy))

    # Without patching with the correct checksum, the function should raise a
    # RuntimeError.
    self.assertRaises(RuntimeError, xeno_canto._load_ebird_taxonomy)

    # A malformed taxonomy (non-unique species codes, for instance) should cause
    # the function to raise an error.
    malformed_csv_str = self.ebird_taxonomy_csv_str.replace(
        'ostric3', 'ostric2')
    mock_run.return_value.stdout = malformed_csv_str.encode('utf-8')
    checksum = hashlib.sha256(
        pd.read_csv(io.StringIO(malformed_csv_str)).to_json().encode(
            'utf-8')).hexdigest()
    with mock.patch.object(
        xeno_canto, '_EBIRD_TAXONOMY_CHECKSUM', new=checksum):
      self.assertRaises(RuntimeError, xeno_canto._load_ebird_taxonomy)

  @parameterized.parameters(
      # A row with 'no_species_code' set to True should not be modified.
      ('fake1,Fakus namus,Fake Name', ',true,Fakus namus,Fake Name,false',
       ',true,Fakus namus,Fake Name,false'),
      # A row with an existing species code should not be modified.
      ('fake1,Fakus namus,Fake Name', 'fake2,false,Fakus namus,Fake Name,false',
       'fake2,false,Fakus namus,Fake Name,false'),
      # The function should prefer matching by scientific name over matching
      # by common name, and it should flag the cleaned up species row for
      # verification.
      ('fake1,Fakus namus,Fake Name\nmock1,Mockus namus,Fake Name',
       ',false,Fakus namus,Fake Name,false',
       'fake1,false,Fakus namus,Fake Name,true'),
      # The function should prefer matching by common name over matching by
      # common name variations, and it should flag the cleaned up species row
      # for verification.
      (('fake1,Mockus namus,Fake Name\n'
        'mock1,Mockus namus,Fake Name (undescribed form)'),
       ',false,Fakus namus,Fake Name,false',
       'fake1,false,Fakus namus,Fake Name,true'),
      # The function should prefer matching by common name variation over
      # matching by scientific name substrings, and it should flag the
      # cleaned up species row for verification.
      (('fake1,Mockus namus,Fake Name (undescribed form)\n'
        'mock1,Mockus fakus namus,Mock name'),
       ',false,Fakus namus,Fake Name,false',
       'fake1,false,Fakus namus,Fake Name,true'),
      ('fake1,Mockus namus,Fake Gray Name',
       ',false,Fakus namus,Fake Grey Name,false',
       'fake1,false,Fakus namus,Fake Grey Name,true'),
      # The function should match by scientific name substring in the last
      # resort, and it should flag the cleaned up species row for
      # verification.
      ('fake1,Fakus mockus namus,Mock Name',
       ',false,Fakus namus,Fake Name,false',
       'fake1,false,Fakus namus,Fake Name,true'),
      # It should however be conservative and avoid modifying the row if there
      # are multiple scientific name substring matches.
      (('fake1,Fakus fakus namus,Mock Name\n'
        'fake2,Fakus mockus namus,Mock Name'),
       ',false,Fakus namus,Fake Name,false',
       ',false,Fakus namus,Fake Name,false'),
  )
  def test_clean_up_species_codes(self, ebird_taxonomy_str, taxonomy_info_str,
                                  expected_taxonomy_info_str):
    ebird_taxonomy_header = 'SPECIES_CODE,SCI_NAME,PRIMARY_COM_NAME'
    taxonomy_info_header = (
        'species_code,no_species_code,Scientific name,Common name,to_verify')

    to_dataframe = lambda h, r: pd.read_csv(io.StringIO('\n'.join((h, r))))
    to_ebird_taxonomy = lambda r: to_dataframe(ebird_taxonomy_header, r)
    to_taxonomy_info = lambda r: to_dataframe(taxonomy_info_header, r)

    self.assertTrue(
        to_taxonomy_info(expected_taxonomy_info_str).equals(
            xeno_canto._clean_up_species_codes(
                to_taxonomy_info(taxonomy_info_str),
                to_ebird_taxonomy(ebird_taxonomy_str))))

  def test_ensure_species_code_uniqueness(self):
    # If there are no collisions, the taxonomy DataFrame should be untouched.
    self.assertTrue(
        xeno_canto._ensure_species_codes_uniqueness(
            self.taxonomy_info, {}).equals(self.taxonomy_info))

    # If there's a disallowed collision, the function should raise an exception.
    non_unique_taxonomy_info = self.taxonomy_info.copy()
    first_code, second_code = non_unique_taxonomy_info['species_code'].to_list(
    )[:2]
    non_unique_taxonomy_info['species_code'] = non_unique_taxonomy_info[
        'species_code'].map(lambda s: (second_code if s == first_code else s))

    self.assertRaises(ValueError, xeno_canto._ensure_species_codes_uniqueness,
                      non_unique_taxonomy_info, {})

    # If there is an allowed collision, the function merge the colliding rows
    # into the first colliding row and drop the others.
    expected_taxonomy_info = non_unique_taxonomy_info.copy()
    colliding_rows = expected_taxonomy_info.iloc[:2]
    expected_taxonomy_info.loc[0] = pd.Series({
        'Common name': ','.join(colliding_rows['Common name']),
        'Scientific name': ','.join(colliding_rows['Scientific name']),
        'No.': 4,
        'No. Back': 2,
        'species_code': second_code,
        'is_insect': False,
        'no_species_code': False,
        'to_verify': False,
    })
    expected_taxonomy_info = expected_taxonomy_info.drop(index=1)

    self.assertTrue(
        xeno_canto._ensure_species_codes_uniqueness(
            non_unique_taxonomy_info, {
                frozenset(non_unique_taxonomy_info.iloc[:2]['Scientific name']):
                    second_code
            }).equals(expected_taxonomy_info))

  def test_break_down_taxonomy(self):
    ebird_taxonomy = pd.read_csv(io.StringIO(self.ebird_taxonomy_csv_str))
    self.assertTrue(
        self.broken_down_taxonomy_info.equals(
            xeno_canto._break_down_taxonomy(self.taxonomy_info,
                                            ebird_taxonomy)))

  @mock.patch.object(xeno_canto.requests, 'Session', autospec=True)
  def test_scrape_xeno_canto_recording_metadata(self, mock_session_cls):
    # Mock requests.Session.get to simulate a Xeno-Canto API call.
    mock_session_cls.return_value.get.side_effect = (
        _make_mock_requests_side_effect(
            self.broken_down_taxonomy_info, wrong_num_recordings=True))

    # We expect a RuntimeError to be raised if the code fails to retrieve the
    # number of recordings declared in the response.
    with self.assertRaises(RuntimeError):
      xeno_canto._scrape_xeno_canto_recording_metadata(
          self.broken_down_taxonomy_info, progress_bar=False)

    # Mock requests.Session.get to simulate a Xeno-Canto API call.
    mock_session_cls.return_value.get.side_effect = (
        _make_mock_requests_side_effect(
            self.broken_down_taxonomy_info, wrong_num_recordings=False))

    # We expect only the first recording to be returned for each species code,
    # since the 'file' is empty for the second one and the third one has a *-nd
    # license.
    species_code_to_recording_metadata = (
        xeno_canto._scrape_xeno_canto_recording_metadata(
            self.broken_down_taxonomy_info, progress_bar=False))
    for i, row in self.broken_down_taxonomy_info.iterrows():
      self.assertListEqual(
          species_code_to_recording_metadata[row['species_code']], [
              xeno_canto.RecordingInfo(
                  xc_id=f'{i:05d}',
                  quality_score={
                      0: 'A',
                      1: '',
                      2: 'no score'
                  }[i],
                  background_species=[[
                      'Struthio molybdophanes', 'Rhea americana',
                      'Struthio camelus'
                  ][i]])
          ])

  @mock.patch.object(xeno_canto.SPARQLWrapper, 'SPARQLWrapper', autospec=True)
  @mock.patch.object(xeno_canto.subprocess, 'run', autospec=True)
  @mock.patch.object(xeno_canto.pd, 'read_html', autospec=True)
  def test_create_taxonomy_info(self, mock_read_html, mock_run,
                                mock_sparql_wrapper):
    name_to_code = dict(
        zip(
            self.broken_down_taxonomy_info['xeno_canto_query'].map(
                lambda n: n.replace(' ', '-')),
            self.broken_down_taxonomy_info['species_code']))
    matches = sorted((c, n) for n, c in name_to_code.items())
    bindings = []
    for n, c in name_to_code.items():
      bindings.append({
          'eBird_taxon_ID': {
              'value': c
          },
          'Xeno_canto_species_ID': {
              'value': n
          }
      })
    mock_sparql_wrapper.return_value.queryAndConvert.return_value = dict(
        results=dict(bindings=bindings))

    mock_run.return_value = mock.MagicMock()
    mock_run.return_value.stdout = self.ebird_taxonomy_csv_str.encode('utf-8')

    self.taxonomy_info['Status'] = None
    mock_read_html.return_value = [
        self.taxonomy_info.drop(columns=[
            'species_code', 'is_insect', 'no_species_code', 'to_verify'
        ])
    ]

    expected = self.broken_down_taxonomy_info.copy()[[
        'No.', 'species_code', 'xeno_canto_query', 'scientific_name', 'species',
        'genus', 'family', 'order', 'common_name'
    ]]

    wikidata_checksum = hashlib.sha256(str(matches).encode('utf-8')).hexdigest()
    with mock.patch.object(
        xeno_canto, '_WIKIDATA_CHECKSUM', new=wikidata_checksum):
      taxonomy = pd.read_csv(io.StringIO(self.ebird_taxonomy_csv_str))
      taxonomy_checksum = hashlib.sha256(
          taxonomy.to_json().encode('utf-8')).hexdigest()
      with mock.patch.object(
          xeno_canto, '_EBIRD_TAXONOMY_CHECKSUM', new=taxonomy_checksum):
        self.assertTrue(
            xeno_canto.create_taxonomy_info(
                xeno_canto.SpeciesMappingConfig()).equals(expected))

  @mock.patch.object(xeno_canto.requests, 'Session', autospec=True)
  def test_retrieve_recording_metadata(self, mock_session_cls):
    # Mock requests.Session.get to simulate a Xeno-Canto API call.
    mock_session_cls.return_value.get.side_effect = (
        _make_mock_requests_side_effect(self.broken_down_taxonomy_info))

    taxonomy_info = self.broken_down_taxonomy_info.drop(columns=[
        'Common name', 'Scientific name', 'No. Back', 'is_insect',
        'no_species_code', 'to_verify'
    ])

    expected = taxonomy_info.copy().drop(columns=['No.'])
    expected['xeno_canto_ids'] = [['00000'], ['00001'], ['00002']]
    expected['xeno_canto_quality_scores'] = [['A'], [''], ['no score']]
    expected['xeno_canto_bg_species_codes'] = [[['ostric3']], [['grerhe1']],
                                               [['ostric2']]]
    self.assertTrue(
        xeno_canto.retrieve_recording_metadata(
            taxonomy_info, progress_bar=False).equals(expected))

    # The function should raise a RuntimeError if the number of foreground
    # recordings declared by `taxonomy_info` is greater than the number of
    # recordings retrieved.
    taxonomy_info_wrong_no = taxonomy_info.copy()
    taxonomy_info_wrong_no['No.'] = 500
    self.assertRaises(
        RuntimeError,
        xeno_canto.retrieve_recording_metadata,
        taxonomy_info_wrong_no,
        progress_bar=False)


if __name__ == '__main__':
  absltest.main()
