create database imdbm template imdb;
\c imdbm

drop table name;
drop table person_info;
drop table role_type;
drop table aka_name;


alter table title drop column episode_of_id;
alter table title drop column imdb_id;
alter table title drop column imdb_index;
alter table title drop column md5sum;

alter table aka_title drop column title;
alter table aka_title drop column imdb_index;
alter table aka_title drop column production_year;
alter table aka_title drop column phonetic_code;
alter table aka_title drop column episode_of_id;
alter table aka_title drop column season_nr;
alter table aka_title drop column episode_nr;
alter table aka_title drop column note;
alter table aka_title drop column md5sum;

alter table keyword drop column phonetic_code;
