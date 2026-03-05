
from SpaceBalls.utils import make_dict_hash_key, make_list_str_key
import sqlite3
import threading

_db_path = '/media/monte_share/EEI_estimations/EEI_estimations_database.db'
#_db_path = 'notebooks/test_database.db'

# Thread/process-local storage for connections
_local = threading.local()

def get_connection():
    """Get or create a connection for the current thread/process."""
    if not hasattr(_local, 'conn') or _local.conn is None:
        _local.conn = sqlite3.connect(_db_path, timeout=10.0, check_same_thread=False)
    return _local.conn

def get_cursor():
    """Get a cursor for the current thread/process."""
    return get_connection().cursor()

# For backward compatibility, initialize module-level connection if in main thread
conn = sqlite3.connect(_db_path, timeout=10.0, check_same_thread=False)
c = conn.cursor()

#def initialize_db(c: sqlite3.Cursor):
def initialize_db():
    _conn = get_connection()
    _c = get_cursor()
    
    _conn.execute("PRAGMA journal_mode=WAL;")

    _c.execute("""CREATE TABLE IF NOT EXISTS config_table (
                estimation_key TEXT PRIMARY KEY,
                EEI_truth_idx INTEGER,
                satellites TEXT,
                accelerometer_errors_key TEXT,
                model_errors_key TEXT
                )""")

    _c.execute("""CREATE TABLE IF NOT EXISTS EEI_postproc_results_table (
                estimation_key TEXT PRIMARY KEY,
                error_RMS_1_day_avg REAL,
                error_RMS_7_day_avg REAL,
                error_RMS_30_day_avg REAL,
                error_RMS_182_day_avg REAL,
                error_RMS_365_day_avg REAL
                )""")

    _c.execute("""CREATE TABLE IF NOT EXISTS accelerometer_errors_table (
                accelerometer_errors_key TEXT PRIMARY KEY
            )""")

    _c.execute("""CREATE TABLE IF NOT EXISTS model_errors_table (
                model_errors_key TEXT PRIMARY KEY
            )""")

    _c.execute("""CREATE TABLE IF NOT EXISTS sample_quality_table(
            constellation_key TEXT PRIMARY KEY
            )""")

    _conn.commit()

initialize_db()

def check_valid_entry(dict_entry):
    bool_not_none = (not(isinstance(dict_entry, dict)) and dict_entry is not None)
    bool_valid_dict = (isinstance(dict_entry, dict) and dict_entry and not(all(v is None for v in dict_entry.values())))
    
    return bool_not_none or bool_valid_dict
                            


#def insert_config_entry(c: sqlite3.Cursor, config_dict: dict):
def insert_config_entry(config_dict: dict, config_dict_hash_key: str):
    _conn = get_connection()
    _c = get_cursor()
    
    with _conn:
        _c.execute("""INSERT INTO config_table (estimation_key, EEI_truth_idx, satellites) VALUES 
                  (:estimation_key, :EEI_truth_idx, :satellites)
                  ON CONFLICT(estimation_key) DO NOTHING;""", # , multi-entry insertion
                  {'estimation_key': config_dict_hash_key, 
                   'EEI_truth_idx': int(''.join(filter(str.isdigit, config_dict["EEI_truth_name"]))), # config_dict["EEI_truth_name"], 
                   'satellites': make_list_str_key(config_dict["satellite_list"])}
                )
        
        if check_valid_entry(config_dict["accelerometer_errors"]): #single entry insertion
            _c.execute(
                "INSERT INTO config_table (accelerometer_errors_key) VALUES (?)",
                (make_dict_hash_key(config_dict["accelerometer_errors"]),)
            )
            # TODO: fill acc errors table
        
        if check_valid_entry(config_dict["model_errors"]): #single entry insertion
            _c.execute(
                "INSERT INTO config_table (model_errors_key) VALUES (?)",
                (make_dict_hash_key(config_dict["model_errors"]),)
            )
            # TODO: fill model errors table

def insert_results_entry(config_dict_hash_key: str, column_name: str, value):
    _conn = get_connection()
    _c = get_cursor()
    
    with _conn:
        _c.execute(f"""INSERT INTO EEI_postproc_results_table (estimation_key, {column_name}) VALUES (?,?)
                      ON CONFLICT(estimation_key) DO NOTHING;""",
                  (config_dict_hash_key, value)
                  )

#def insert_sample_quality_entry(c: sqlite3.Cursor):
def insert_sample_quality_entry():
    pass


#def check_if_entry_exists(c: sqlite3.Cursor, table_name, column_name, entry):


def check_if_entry_exists(table_name, column_name, entry):
    _c = get_cursor()
    _c.execute(f"SELECT 1 FROM {table_name} where {column_name} = ?", (entry,))
    return (_c.fetchone() is not None)


def check_if_entry_is_not_none(table_name, primary_key_column, primary_key_value, column_name):
    _c = get_cursor()
    _c.execute(
        f"SELECT 1 FROM {table_name} WHERE {primary_key_column} = ? AND {column_name} IS NOT NULL LIMIT 1",
        (primary_key_value,)
    )
    return _c.fetchone() is not None


# just a test, deprecated?
def retrieve_column_from_condition(table_name, out_column_name, condition_column_name, 
                                      condition_value, condtion_operator='='):
    _c = get_cursor()
    _c.execute(
        f"SELECT {out_column_name} FROM {table_name} WHERE {condition_column_name}{condtion_operator}{str(condition_value)}"
    )
    all_rows = _c.fetchall()
    return [t[0] for t in all_rows]



def retrieve_column(table_name, column_name):
    _c = get_cursor()
    _c.execute(
        f"SELECT * FROM {table_name}"
    )
    all_rows = _c.fetchall()

    return all_rows

def retrieve_2_table_columns_from_1_table_condition(
        condition_table, other_table, common_key_name,
                                    results_column, config_column,
                                    condition_column, condition_value,
                                    condition_operator='='):
    _c = get_cursor()
    _c.execute(f"""
        SELECT r.{results_column}, c.{config_column}
        FROM {condition_table} r
        JOIN {other_table} c ON r.{common_key_name} = c.{common_key_name}
        WHERE r.{condition_column} {condition_operator} ?
    """, (condition_value,))
    
    rows = _c.fetchall()
    return rows
        


def retrieve_columns_from_condition(results_table, config_table,
                                    results_column, config_column,
                                    condition_column, condition_value,
                                    common_key_name,
                                    condition_operator='='):
    _c = get_cursor()
    _c.execute(f"""
        SELECT c.{common_key_name}, r.{results_column}, c.{config_column}
        FROM {results_table} r
        JOIN {config_table} c ON r.{common_key_name} = c.{common_key_name}
        WHERE r.{condition_column} {condition_operator} ?
    """, (condition_value,))
    
    rows = _c.fetchall()
    return rows
        