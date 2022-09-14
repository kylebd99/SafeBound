create database imdblightranges template imdb;
\c imdblightranges

drop table keyword;
drop table company_type;
drop table aka_name;
drop table aka_title;
drop table comp_cast_type;
drop table char_name;
drop table company_name;
drop table complete_cast;
drop table info_type;
drop table kind_type;
drop table link_type;
drop table name;
drop table person_info;
drop table role_type;
drop table movie_link;


alter table cast_info drop column person_role_id;
alter table cast_info drop column note;

alter table movie_companies drop note;

alter table movie_info drop note;
alter table movie_info drop info;

alter table movie_info_idx drop column info;
alter table movie_info_idx drop column note;

alter table title drop column episode_of_id;

