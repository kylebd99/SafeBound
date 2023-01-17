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

def load_tpcds(size="L"):
    dbgen_version = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/dbgen_version.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=["dv_version", "dv_create_date", "dv_create_time", "dv_cmdline_args"], low_memory=False)
    customer_address = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/customer_address.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=["ca_address_sk", "ca_address_id", "ca_street_number", "ca_street_name", 
                              "ca_street_type", "ca_suite_number", "ca_city", "ca_county", "ca_state", "ca_zip",
                              "ca_country", "ca_gmt_offset", "ca_location_type"], low_memory=False)
    customer_demographics = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/customer_demographics.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=["cd_demo_sk", "cd_gender", "cd_marital_status", "cd_education_status", "cd_purchase_estimate",
                              "cd_credit_rating", "cd_dep_count", "cd_dep_employed_count", "cd_dep_college_count"], low_memory=False)
    date_dim = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/date_dim.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=["d_date_sk", "d_date_id", "d_date", "d_month_seq", "d_week_seq", "d_quarter_seq",
                                "d_year", "d_dow", "d_moy", "d_dom", "d_qoy", "d_fy_year", "d_fy_quarter_seq", "d_fy_week_seq", "d_day_name",
                                "d_quarter_name", "d_holiday", "d_weekend", "d_following_holiday", "d_first_dom", "d_last_dom", "d_same_day_ly",
                                "d_same_day_lq", "d_current_day", "d_current_week", "d_current_month", "d_current_quarter", "d_current_year"], low_memory=False)
    warehouse = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/warehouse.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=["w_warehouse_sk", "w_warehouse_id", "w_warehouse_name", "w_warehouse_sq_ft", "w_street_number", "w_street_name",
                                "w_street_type", "w_suite_number", "w_city", "w_county", "w_state", "w_zip", "w_country", "w_gmt_offset"], low_memory=False)
    ship_mode = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/ship_mode.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=['sm_ship_mode_sk', 'sm_ship_mode_id',  'sm_type',  'sm_code',  'sm_carrier', 'sm_contract'], low_memory=False)
    time_dim = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/time_dim.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=['t_time_sk', 't_time_id', 't_time',  't_hour', 't_minute', 't_second', 't_am_pm', 't_shift', 't_sub_shift', 't_meal_time'], low_memory=False)
    reason = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/reason.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=['r_reason_sk', 'r_reason_id', 'r_reason_desc'], low_memory=False)
    income_band = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/income_band.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=['ib_income_band_sk', 'ib_lower_bound', 'ib_upper_bound'], low_memory=False)
    item = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/item.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=['i_item_sk', 'i_item_id', 'i_rec_start_date', 'i_rec_end_date', 'i_item_desc', 'i_current_price',
                                'i_wholesale_cost', 'i_brand_id', 'i_brand', 'i_class_id', 'i_class', 'i_category_id', 'i_category', 'i_manufact_id', 'i_manufact',
                                'i_size', 'i_formulation', 'i_color', 'i_units', 'i_container', 'i_manager_id', 'i_product_name'], low_memory=False)
    store = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/store.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=['s_store_sk', 's_store_id', 's_rec_start_date', 's_rec_end_date', 's_closed_date_sk', 's_store_name',
                                 's_number_employees', 's_floor_space', 's_hours', 's_manager', 's_market_id', 's_geography_class', 's_market_desc', 's_market_manager',
                                 's_division_id', 's_division_name', 's_company_id', 's_company_name', 's_street_number', 's_street_name', 's_street_type',
                                 's_suite_number', 's_city', 's_county', 's_state', 's_zip', 's_country', 's_gmt_offset', 's_tax_precentage'], low_memory=False)
    call_center = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/call_center.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=['cc_call_center_sk', 'cc_call_center_id', 'cc_rec_start_date',
                                'cc_rec_end_date', 'cc_closed_date_sk', 'cc_open_date_sk', 'cc_name', 'cc_class', 'cc_employees',
                                'cc_sq_ft', 'cc_hours', 'cc_manager', 'cc_mkt_id', 'cc_mkt_class', 'cc_mkt_desc',
                                'cc_market_manager', 'cc_division', 'cc_division_name', 'cc_company', 'cc_company_name', 'cc_street_number', 'cc_street_name', 'cc_street_type',
                                'cc_suite_number', 'cc_city', 'cc_county', 'cc_state', 'cc_zip', 'cc_country', 'cc_gmt_offset', 'cc_tax_percentage'], low_memory=False)
    customer = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/customer.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=["c_customer_sk", "c_customer_id", "c_current_cdemo_sk", "c_current_hdemo_sk", "c_current_addr_sk", "c_first_shipto_date_sk", "c_first_sales_date_sk",
                              "c_salutation", "c_first_name", "c_last_name", "c_preferred_cust_flag", "c_birth_day", "c_birth_month", "c_birth_year", "c_birth_country", "c_login", 
                              "c_email_address", "c_last_review_date"], low_memory=False)
    web_site = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/web_site.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=["web_site_sk", "web_site_id", "web_rec_start_date", "web_rec_end_date", "web_name", "web_open_date_sk", "web_close_date_sk", "web_class", "web_manager", 
                              "web_mkt_id", "web_mkt_class", "web_mkt_desc", "web_market_manager", "web_company_id", "web_company_name", "web_street_number", "web_street_name", 
                              "web_street_type", "web_suite_number", "web_city", "web_county", "web_state", "web_zip", "web_country", "web_gmt_offset", "web_tax_percentage"], 
                           low_memory=False)
    store_returns = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/store_returns.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=["sr_returned_date_sk", "sr_return_time_sk", "sr_item_sk", "sr_customer_sk", "sr_cdemo_sk", "sr_hdemo_sk", "sr_addr_sk", "sr_store_sk", "sr_reason_sk", 
                              "sr_ticket_number", "sr_return_quantity", "sr_return_amt", "sr_return_tax", "sr_return_amt_inc_tax", "sr_fee", "sr_return_ship_cost", "sr_refunded_cash", 
                              "sr_reversed_charge", "sr_store_credit", "sr_net_loss"], 
                           low_memory=False)
    household_demographics = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/household_demographics.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=["hd_demo_sk", "hd_income_band_sk", "hd_buy_potential", "hd_dep_count", "hd_vehicle_count"], 
                           low_memory=False)
    web_page = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/web_page.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=["wp_web_page_sk", "wp_web_page_id", "wp_rec_start_date", "wp_rec_end_date", "wp_creation_date_sk", "wp_access_date_sk", "wp_autogen_flag",
                              "wp_customer_sk", "wp_url", "wp_type", "wp_char_count", "wp_link_count", "wp_image_count", "wp_max_ad_count"], 
                           low_memory=False)
    promotion = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/promotion.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=["p_promo_sk", "p_promo_id", "p_start_date_sk", "p_end_date_sk", "p_item_sk", "p_cost", "p_response_target", "p_promo_name", "p_channel_dmail", 
                              "p_channel_email", "p_channel_catalog", "p_channel_tv", "p_channel_radio", "p_channel_press", "p_channel_event", "p_channel_demo", "p_channel_details", 
                              "p_purpose", "p_discount_active"], 
                           low_memory=False)
    catalog_page = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/catalog_page.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=["cp_catalog_page_sk", "cp_catalog_page_id", "cp_start_date_sk", "cp_end_date_sk", "cp_department", "cp_catalog_number", "cp_catalog_page_number",
                              "cp_description", "cp_type"], 
                           low_memory=False)
    inventory = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/inventory.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=["inv_date_sk", "inv_item_sk", "inv_warehouse_sk", "inv_quantity_on_hand"], 
                           low_memory=False)
    catalog_returns = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/catalog_returns.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=["cr_returned_date_sk", "cr_returned_time_sk", "cr_item_sk", "cr_refunded_customer_sk", "cr_refunded_cdemo_sk", "cr_refunded_hdemo_sk", "cr_refunded_addr_sk", 
                              "cr_returning_customer_sk", "cr_returning_cdemo_sk", "cr_returning_hdemo_sk", "cr_returning_addr_sk", "cr_call_center_sk", "cr_catalog_page_sk", 
                              "cr_ship_mode_sk", "cr_warehouse_sk", "cr_reason_sk", "cr_order_number", "cr_return_quantity", "cr_return_amount", "cr_return_tax", "cr_return_amt_inc_tax", 
                              "cr_fee", "cr_return_ship_cost", "cr_refunded_cash", "cr_reversed_charge", "cr_store_credit", "cr_net_loss"], 
                           low_memory=False)
    web_returns = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/web_returns.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=["wr_returned_date_sk", "wr_returned_time_sk", "wr_item_sk", "wr_refunded_customer_sk", "wr_refunded_cdemo_sk", "wr_refunded_hdemo_sk", "wr_refunded_addr_sk",
                              "wr_returning_customer_sk", "wr_returning_cdemo_sk", "wr_returning_hdemo_sk", "wr_returning_addr_sk", "wr_web_page_sk", "wr_reason_sk", "wr_order_number", 
                              "wr_return_quantity", "wr_return_amt", "wr_return_tax", "wr_return_amt_inc_tax", "wr_fee", "wr_return_ship_cost", "wr_refunded_cash", "wr_reversed_charge", 
                              "wr_account_credit", "wr_net_loss"], 
                           low_memory=False)
    web_sales = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/web_sales.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=["ws_sold_date_sk", "ws_sold_time_sk", "ws_ship_date_sk", "ws_item_sk", "ws_bill_customer_sk", "ws_bill_cdemo_sk", "ws_bill_hdemo_sk", "ws_bill_addr_sk", 
                              "ws_ship_customer_sk", "ws_ship_cdemo_sk", "ws_ship_hdemo_sk", "ws_ship_addr_sk", "ws_web_page_sk", "ws_web_site_sk", "ws_ship_mode_sk", "ws_warehouse_sk",
                              "ws_promo_sk", "ws_order_number", "ws_quantity", "ws_wholesale_cost", "ws_list_price", "ws_sales_price", "ws_ext_discount_amt", "ws_ext_sales_price", 
                              "ws_ext_wholesale_cost", "ws_ext_list_price", "ws_ext_tax", "ws_coupon_amt", "ws_ext_ship_cost", "ws_net_paid", "ws_net_paid_inc_tax", "ws_net_paid_inc_ship", 
                              "ws_net_paid_inc_ship_tax", "ws_net_profit"], 
                           low_memory=False)
    catalog_sales = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/catalog_sales.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=["cs_sold_date_sk", "cs_sold_time_sk", "cs_ship_date_sk", "cs_bill_customer_sk", "cs_bill_cdemo_sk", "cs_bill_hdemo_sk", "cs_bill_addr_sk", 
                              "cs_ship_customer_sk", "cs_ship_cdemo_sk", "cs_ship_hdemo_sk", "cs_ship_addr_sk", "cs_call_center_sk", "cs_catalog_page_sk", "cs_ship_mode_sk", 
                              "cs_warehouse_sk", "cs_item_sk", "cs_promo_sk", "cs_order_number", "cs_quantity", "cs_wholesale_cost", "cs_list_price", "cs_sales_price", "cs_ext_discount_amt",
                              "cs_ext_sales_price", "cs_ext_wholesale_cost", "cs_ext_list_price", "cs_ext_tax", "cs_coupon_amt", "cs_ext_ship_cost", "cs_net_paid", "cs_net_paid_inc_tax", 
                              "cs_net_paid_inc_ship", "cs_net_paid_inc_ship_tax", "cs_net_profit"], 
                           low_memory=False)
    store_sales = pd.read_csv(rootDirectory + "Data/TPCDS-" + size + "/store_sales.dat", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                       names=["ss_sold_date_sk", "ss_sold_time_sk", "ss_item_sk", "ss_customer_sk", "ss_cdemo_sk", "ss_hdemo_sk", "ss_addr_sk", "ss_store_sk", "ss_promo_sk", 
                              "ss_ticket_number", "ss_quantity", "ss_wholesale_cost", "ss_list_price", "ss_sales_price", "ss_ext_discount_amt", "ss_ext_sales_price", "ss_ext_wholesale_cost",
                              "ss_ext_list_price", "ss_ext_tax", "ss_coupon_amt", "ss_net_paid", "ss_net_paid_inc_tax", "ss_net_profit"], 
                           low_memory=False)
    
    return {'dbgen_version': dbgen_version,
            'customer_address':customer_address,
            'customer_demographics':customer_demographics,
            'date_dim':date_dim,
            'warehouse':warehouse,
            'ship_mode':ship_mode,
            'time_dim':time_dim,
            'reason':reason,
            'income_band':income_band,
            'item':item,
            'store':store,
            'call_center':call_center,
            'customer':customer,
            'web_site':web_site,
            'store_returns':store_returns,
            'household_demographics':household_demographics,
            'web_page':web_page,
            'promotion':promotion,
            'catalog_page':catalog_page,
            'inventory':inventory,
            'catalog_returns':catalog_returns,
            'web_returns':web_returns,
            'web_sales':web_sales,
            'catalog_sales':catalog_sales,
            'store_sales':store_sales,
           }

def load_tpch(size="1"):
    
    nation = pd.read_csv(rootDirectory + "Data/TPCH-" + size + "/nation.tbl", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                                   names=["n_nationkey", "n_name", "n_regionkey" , "n_comment", "n_dummy"], low_memory=False)
    region = pd.read_csv(rootDirectory + "Data/TPCH-" + size + "/region.tbl", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                                   names=["r_regionkey", "r_name", "r_comment", "r_dummy"], low_memory=False)
    supplier = pd.read_csv(rootDirectory + "Data/TPCH-" + size + "/supplier.tbl", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                                   names=["s_suppkey", "s_name", "s_address", "s_nationkey",  "s_phone", "s_acctbal", "s_comment", "s_dummy"], low_memory=False)
    customer = pd.read_csv(rootDirectory + "Data/TPCH-" + size + "/customer.tbl", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                                   names=["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", "c_acctbal", "c_mktsegment", "c_comment", "c_dummy"], low_memory=False)
    part = pd.read_csv(rootDirectory + "Data/TPCH-" + size + "/part.tbl", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                                   names=["p_partkey", "p_name", "p_mfgr", "p_brand", "p_type", "p_size", "p_container", "p_retailprice", "p_comment", "p_dummy"], low_memory=False)
    partsupp = pd.read_csv(rootDirectory + "Data/TPCH-" + size + "/partsupp.tbl", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                                   names=["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment", "ps_dummy"], low_memory=False)
    orders = pd.read_csv(rootDirectory + "Data/TPCH-" + size + "/orders.tbl", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                                   names=["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate", "o_orderpriority", "o_clerk", "o_shippriority", "o_comment",
                                          "o_dummy"], low_memory=False)
    lineitem = pd.read_csv(rootDirectory + "Data/TPCH-" + size + "/lineitem.tbl", quotechar = "\"", escapechar="\\", delimiter="|",index_col=False,
                                   names=["l_orderkey", "l_partkey", "l_suppkey","l_linenumber", "l_quantity", "l_extendedprice", "l_discount", "l_tax", "l_returnflag", 
                                          "l_linestatus", "l_shipdate", "l_commitdate", "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment", "l_dummy"], low_memory=False)
    return {'nation': nation,
            'region': region,
            'supplier':supplier,
            'customer':customer,
            'part':part,
            'partsupp':partsupp,
            'orders':orders,
            'lineitem':lineitem
           }
    