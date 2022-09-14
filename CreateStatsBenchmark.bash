createdb stats
psql stats -f Data/Stats/stats.sql
psql stats -f Data/Stats/stats_index.sql
psql stats -f Data/Stats/stats_load.sql
