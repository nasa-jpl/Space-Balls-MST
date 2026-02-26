
from SpaceBalls.utils import make_dict_hash_key, make_list_str_key
import sqlite3
conn = sqlite3.connect('/media/monte_share/EEI_estimations/EEI_estimations_database.db')
#conn = sqlite3.connect('notebooks/test_database.db')
c = conn.cursor()

#def initialize_db(c: sqlite3.Cursor):
def initialize_db():

    c.execute("""CREATE TABLE IF NOT EXISTS config_table (
                estimation_key TEXT PRIMARY KEY,
                EEI_truth_idx INTEGER,
                satellites TEXT,
                accelerometer_errors_key TEXT,
                model_errors_key TEXT
                )""")

    c.execute("""CREATE TABLE IF NOT EXISTS EEI_postproc_results_table (
                estimation_key TEXT PRIMARY KEY,
                error_RMS_1_day_avg REAL,
                error_RMS_7_day_avg REAL,
                error_RMS_30_day_avg REAL,
                error_RMS_182_day_avg REAL,
                error_RMS_365_day_avg REAL
                )""")

    c.execute("""CREATE TABLE IF NOT EXISTS accelerometer_errors_table (
                accelerometer_errors_key TEXT PRIMARY KEY
            )""")

    c.execute("""CREATE TABLE IF NOT EXISTS model_errors_table (
                model_errors_key TEXT PRIMARY KEY
            )""")

    c.execute("""CREATE TABLE IF NOT EXISTS sample_quality_table(
            constellation_key TEXT PRIMARY KEY
            )""")

initialize_db()

def check_valid_entry(dict_entry):
    bool_not_none = (not(isinstance(dict_entry, dict)) and dict_entry is not None)
    bool_valid_dict = (isinstance(dict_entry, dict) and dict_entry and not(all(v is None for v in dict_entry.values())))
    
    return bool_not_none or bool_valid_dict
                            


#def insert_config_entry(c: sqlite3.Cursor, config_dict: dict):
def insert_config_entry(config_dict: dict, config_dict_hash_key: str):
    with conn:
        c.execute("""INSERT INTO config_table (estimation_key, EEI_truth_idx, satellites) VALUES 
                  (:estimation_key, :EEI_truth_idx, :satellites)
                  ON CONFLICT(estimation_key) DO NOTHING;""", # , multi-entry insertion
                  {'estimation_key': config_dict_hash_key, 
                   'EEI_truth_idx': int(''.join(filter(str.isdigit, config_dict["EEI_truth_name"]))), # config_dict["EEI_truth_name"], 
                   'satellites': make_list_str_key(config_dict["satellite_list"])}
                )
        
        if check_valid_entry(config_dict["accelerometer_errors"]): #single entry insertion
            c.execute(
                "INSERT INTO config_table (accelerometer_errors_key) VALUES (?)",
                (make_dict_hash_key(config_dict["accelerometer_errors"]),)
            )
            # TODO: fill acc errors table
        
        if check_valid_entry(config_dict["model_errors"]): #single entry insertion
            c.execute(
                "INSERT INTO config_table (model_errors_key) VALUES (?)",
                (make_dict_hash_key(config_dict["model_errors"]),)
            )
            # TODO: fill model errors table

def insert_results_entry(config_dict_hash_key: str, column_name: str, value):
    with conn:
        c.execute(f"""INSERT INTO EEI_postproc_results_table (estimation_key, {column_name}) VALUES (?,?)
                      ON CONFLICT(estimation_key) DO NOTHING;""",
                  (config_dict_hash_key, value)
                  )

#def insert_sample_quality_entry(c: sqlite3.Cursor):
def insert_sample_quality_entry():
    pass


#def check_if_entry_exists(c: sqlite3.Cursor, table_name, column_name, entry):


def check_if_entry_exists(table_name, column_name, entry): 
    
    c.execute(f"SELECT 1 FROM {table_name} where {column_name} = ?", (entry,))
    return (c.fetchone() is not None)


def check_if_entry_is_not_none(table_name, primary_key_column, primary_key_value, column_name):
    c.execute(
        f"SELECT 1 FROM {table_name} WHERE {primary_key_column} = ? AND {column_name} IS NOT NULL LIMIT 1",
        (primary_key_value,)
    )
    return c.fetchone() is not None
        