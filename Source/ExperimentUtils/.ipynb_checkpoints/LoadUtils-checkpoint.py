import pandas as pd
import os
rootDirectory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) +'/'



def load_imdb():
    aka_name = pd.read_csv(rootDirectory + "Data/IMDB/aka_name.csv", quotechar = "\"", escapechar="\\", 
                       names=["id", "person_id", "name", "imdb_index", "name_pcode_cf", "name_pcode_nf",
                              "surname_pcode", "md5sum"], low_memory=False)
    aka_title = pd.read_csv(rootDirectory + "Data/IMDB/aka_title.csv", quotechar = "\"", escapechar="\\",
                           names=["id", "movie_id", "title", "imdb_index", "kind_id", "production_year",
                                  "phonetic_code", "episode_of_id", "season_nr", "episode_nr", "note", "md5sum"], low_memory=False)
    cast_info = pd.read_csv(rootDirectory + "Data/IMDB/cast_info.csv", quotechar = "\"", escapechar="\\",
                           names=["id", "person_id", "movie_id", "person_role_id", "note", "nr_order", "role_id"], low_memory=False)
    char_name = pd.read_csv(rootDirectory + "Data/IMDB/char_name.csv", quotechar = "\"", escapechar="\\",
                           names=["id", "name", "imdb_index", "imdb_id", "name_pcode_nf", "surname_pcode", "md5sum"], low_memory=False)
    comp_cast_type = pd.read_csv(rootDirectory + "Data/IMDB/comp_cast_type.csv", quotechar = "\"", escapechar="\\",
                           names=["id","kind"], low_memory=False)
    company_name = pd.read_csv(rootDirectory + "Data/IMDB/company_name.csv", quotechar = "\"", escapechar="\\",
                           names=["id", "name", "country_code", "imdb_id", "name_pcode_nf", "name_pcode_sf", "md5sum"], low_memory=False)
    company_type = pd.read_csv(rootDirectory + "Data/IMDB/company_type.csv", quotechar = "\"", escapechar="\\",
                           names=["id", "kind"], low_memory=False)
    complete_cast = pd.read_csv(rootDirectory + "Data/IMDB/complete_cast.csv", quotechar = "\"", escapechar="\\",
                           names=["id", "movie_id", "subject_id", "status_id"], low_memory=False)
    info_type = pd.read_csv(rootDirectory + "Data/IMDB/info_type.csv", quotechar = "\"", escapechar="\\",
                           names=["id", "info"], low_memory=False)
    keyword = pd.read_csv(rootDirectory + "Data/IMDB/keyword.csv", quotechar = "\"", escapechar="\\",
                           names=["id", "keyword", "phonetic_code"], low_memory=False)
    kind_type = pd.read_csv(rootDirectory + "Data/IMDB/kind_type.csv", quotechar = "\"", escapechar="\\",
                           names=["id", "kind"], low_memory=False)
    link_type = pd.read_csv(rootDirectory + "Data/IMDB/link_type.csv", quotechar = "\"", escapechar="\\",
                           names=["id", "link"], low_memory=False)
    movie_companies = pd.read_csv(rootDirectory + "Data/IMDB/movie_companies.csv", quotechar = "\"", escapechar="\\",
                           names=["id", "movie_id", "company_id", "company_type_id", "note"], low_memory=False)
    movie_info = pd.read_csv(rootDirectory + "Data/IMDB/movie_info.csv", quotechar = "\"", escapechar="\\",
                           names=["id", "movie_id", "info_type_id", "info", "note"], low_memory=False)
    movie_info_idx = pd.read_csv(rootDirectory + "Data/IMDB/movie_info_idx.csv", quotechar = "\"", escapechar="\\",
                           names=["id", "movie_id", "info_type_id", "info", "note"], low_memory=False)
    movie_keyword = pd.read_csv(rootDirectory + "Data/IMDB/movie_keyword.csv", quotechar = "\"", escapechar="\\",
                           names=["id", "movie_id", "keyword_id"], low_memory=False)
    movie_link = pd.read_csv(rootDirectory + "Data/IMDB/movie_link.csv", quotechar = "\"", escapechar="\\",
                           names=["id", "movie_id", "linked_movie_id", "link_type_id"], low_memory=False)
    name = pd.read_csv(rootDirectory + "Data/IMDB/name.csv", quotechar = "\"", escapechar="\\",
                           names=["id", "name", "imdb_index", "imdb_id", "gender", "name_pcode_cf",
                                  "name_pcode_nf", "surname_pcode", "md5sum"], low_memory=False)
    person_info = pd.read_csv(rootDirectory + "Data/IMDB/person_info.csv", quotechar = "\"", escapechar="\\",
                           names=["id", "person_id", "info_type_id", "info", "note"], low_memory=False)
    role_type = pd.read_csv(rootDirectory + "Data/IMDB/role_type.csv", quotechar = "\"", escapechar="\\",
                           names=["id", "role"], low_memory=False)
    title = pd.read_csv(rootDirectory + "Data/IMDB/title.csv", quotechar = "\"", escapechar="\\",
                           names=["id", "title", "imdb_index", "kind_id", "production_year", "imdb_id",
                                  "phonetic_code", "episode_of_id", "season_nr", "episode_nr", "series_years", "md5sum"],
                       low_memory=False)
    return {'aka_name': aka_name,
            'aka_title' : aka_title, 
            'cast_info' : cast_info,
            'char_name' : char_name,
            'comp_cast_type' : comp_cast_type,
            'company_name' : company_name,
            'company_type' : company_type,
            'complete_cast' : complete_cast,  
            'info_type' : info_type,
            'keyword' : keyword,
            'kind_type' : kind_type,
            'link_type' : link_type,
            'movie_companies' : movie_companies,
            'movie_info' : movie_info,
            'movie_info_idx' : movie_info_idx,
            'movie_keyword' : movie_keyword,
            'movie_link' : movie_link,
            'name' : name,
            'person_info' : person_info,
            'role_type' : role_type,
            'title' : title}

def load_stats():
    badges = pd.read_csv(rootDirectory + "Data/Stats/stats_simplified/badges.csv")
    comments = pd.read_csv(rootDirectory + "Data/Stats/stats_simplified/comments.csv")
    postHistory = pd.read_csv(rootDirectory + "Data/Stats/stats_simplified/postHistory.csv")
    postLinks = pd.read_csv(rootDirectory + "Data/Stats/stats_simplified/postLinks.csv")
    posts = pd.read_csv(rootDirectory + "Data/Stats/stats_simplified/posts.csv")
    tags = pd.read_csv(rootDirectory + "Data/Stats/stats_simplified/tags.csv")
    users = pd.read_csv(rootDirectory + "Data/Stats/stats_simplified/users.csv")
    votes = pd.read_csv(rootDirectory + "Data/Stats/stats_simplified/votes.csv")
    return {'badges': badges,
            'comments' : comments,
            'postHistory' : postHistory,
            'postLinks' : postLinks,
            'posts' : posts,
            'tags' : tags,
            'users' : users,
            'votes' : votes}
