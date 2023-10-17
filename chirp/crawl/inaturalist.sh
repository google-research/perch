#!/bin/bash
# inaturalist.sh gbif-observations-dwca.zip inaturalist.csv

# We speed up the SQLite import by removing all non-bird and non-sound data.
# There are approx. 72M lines in the first file, and about 118M in the second.
# This should take about ~5-10 minutes to run on a quick work station.
FILTERED_OBSERVATIONS=$(mktemp)
>&2 echo "Storing filtered observations in ${FILTERED_OBSERVATIONS}"
unzip -p $1 observations.csv | pv -l | awk 'NR==1 || /Aves/' > ${FILTERED_OBSERVATIONS}

FILTERED_MEDIA=$(mktemp)
>&2 echo "Storing filtered media in ${FILTERED_MEDIA}"
unzip -p $1 media.csv | pv -l | awk 'NR==1 || /Sound/' > ${FILTERED_MEDIA}

# Import the files into SQLite to do the join; select only the recordings that
# have an identified species and then save the output to CSV. This takes another
# ~5-10 minutes or so.
sqlite3 <<EOF
.mode csv
.import ${FILTERED_OBSERVATIONS} observations
.import ${FILTERED_MEDIA} media
.headers on
.output ${2}
SELECT o.id, o.taxonID, o.scientificName, m.identifier
FROM observations o INNER JOIN media m ON o.id = m.id
WHERE o.class = 'Aves' AND m.type = 'Sound' AND o.taxonRank = 'species';
EOF