wget http://homepages.cwi.nl/~boncz/job/imdb.tgz --directory-prefix Data/IMDB 
tar zxvf Data/IMDB/imdb.tgz -C Data/IMDB
createdb imdb
psql imdb -f Data/IMDB/schema.sql
psql imdb -f Data/IMDB/imdb_create.sql
psql imdb -f Data/IMDB/fkindexes.sql
psql postgres -f Data/IMDB/CreateJOBLightDB.sql
psql postgres -f Data/IMDB/CreateJOBLightRangesDB.sql
psql postgres -f Data/IMDB/CreateJOBMDB.sql
