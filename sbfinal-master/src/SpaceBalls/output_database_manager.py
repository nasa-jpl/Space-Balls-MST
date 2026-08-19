import numpy as np
from SpaceBalls.utils import make_dict_hash_key, make_list_str_key
import sqlite3
conn = sqlite3.connect('/media/monte_share/EEI_estimations/EEI_estimations_database.db')
#conn = sqlite3.connect('notebooks/test_database.db')
c = conn.cursor()

#def initialize_db(c: sqlite3.Cursor):
def initialize_db():

    conn.execute("PRAGMA journal_mode=WAL;")

    c.execute("""CREATE TABLE IF NOT EXISTS config_table (
                estimation_key TEXT PRIMARY KEY,
                EEI_truth_idx INTEGER,
                satellites TEXT,
                accelerometer_errors_key TEXT,
                model_errors_key TEXT,
                stacking_frame TEXT,
                grid_name TEXT,
                method TEXT,
                fill_zeroes_method TEXT,
                avg_method TEXT
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
        recomp_flag = 1 if (config_dict.get('meas_source') is not None) else None
        c.execute("""INSERT INTO config_table (estimation_key, EEI_truth_idx, satellites,
                  stacking_frame, grid_name, method, fill_zeroes_method, avg_method, n_sb, use_recomp_meas) VALUES 
                  (:estimation_key, :EEI_truth_idx, :satellites, :frame, :grid, :method, :method_0, :method_avg, :n_sb, :recomp_flag)
                  ON CONFLICT(estimation_key) DO NOTHING;""", # , multi-entry insertion
                  {'estimation_key': config_dict_hash_key, 
                   'EEI_truth_idx': int(''.join(filter(str.isdigit, config_dict["EEI_truth_name"]))), # config_dict["EEI_truth_name"], 
                   'satellites': make_list_str_key(config_dict["satellite_list"]),
                   'frame': config_dict['stacking_frame'],
                   'grid': config_dict['grid_name'],
                   'method': config_dict['method'],
                   'method_0': config_dict['fill_zeroes_method'],
                   'method_avg': config_dict['avg_method'],
                   'n_sb': len(config_dict["satellite_list"]),
                   'recomp_flag': recomp_flag}
                )
        # if config_dict.get('meas_source') is not None:
        #     c.execute("""INSERT INTO config_table (estimation_key, use_recomp_meas) VALUES (:key, :recomp_flag)
        #               ON CONFLICT(estimation_key) DO NOTHING;""",
        #               {'key': config_dict_hash_key,
        #                'recomp_flag': 1}
        #     )
        
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
        c.execute(f"""INSERT INTO EEI_postproc_results_table (estimation_key, {column_name}) 
                VALUES (?,?)
                ON CONFLICT(estimation_key) DO UPDATE SET {column_name}=excluded.{column_name}""",
            (config_dict_hash_key, value)
            )
        # c.execute(f"""INSERT INTO EEI_postproc_results_table (estimation_key, {column_name}) VALUES (?,?)""",
        #               #ON CONFLICT(estimation_key) DO NOTHING;""",
        #           (config_dict_hash_key, value)
        #           )

def insert_n_sb():
    with conn:
        sb_rows = retrieve_table_rows('config_table')
        for row in sb_rows:
            pk = row[0]
            const = row[2]
            n_sb = len(const.split('|'))
            c.execute("""INSERT INTO config_table (estimation_key, n_sb)
                      VALUES (?, ?)
                      ON CONFLICT(estimation_key) DO UPDATE SET n_sb=excluded.n_sb""",
                      (pk, n_sb))
        print("done")
            

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


# just a test, deprecated?
def retrieve_column_from_condition(table_name, out_column_name, condition_column_name, 
                                      condition_value, condtion_operator='='):
    c.execute(
        f"SELECT {out_column_name} FROM {table_name} WHERE {condition_column_name}{condtion_operator}{str(condition_value)}"
    )
    all_rows = c.fetchall()
    return [t[0] for t in all_rows]

def retrieve_table_rows(table_name):
    c.execute(
        f"SELECT * FROM {table_name}"
    )
    all_rows = c.fetchall()

    return all_rows

def retrieve_column(table_name, column_name):
    c.execute(
        f"SELECT {column_name} FROM {table_name}"
    )
    all_rows = c.fetchall()

    return all_rows

def retrieve_2_table_columns_from_1_table_condition(
        condition_table, other_table, common_key_name,
                                    results_column, config_column,
                                    condition_column, condition_value,
                                    condition_operator='='):

    c.execute(f"""
        SELECT r.{results_column}, c.{config_column}
        FROM {condition_table} r
        JOIN {other_table} c ON r.{common_key_name} = c.{common_key_name}
        WHERE r.{condition_column} {condition_operator} ?
    """, (condition_value,))
    
    rows = c.fetchall()
    return rows
        


def retrieve_columns_from_condition(results_table, config_table,
                                    results_column, config_column,
                                    condition_column, condition_value,
                                    common_key_name,
                                    condition_operator='='):

    c.execute(f"""
        SELECT c.{common_key_name}, r.{results_column}, c.{config_column}
        FROM {results_table} r
        JOIN {config_table} c ON r.{common_key_name} = c.{common_key_name}
        WHERE r.{condition_column} {condition_operator} ?
    """, (condition_value,))
    
    rows = c.fetchall()
    return rows
    

def get_primary_keys(filters, table_name="config_table", pk_column="estimation_key"):
    

    """
    Retrieve primary keys from a table based on column filters.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    table_name : str
        Name of the table.
    filters : dict
        Dictionary of {column_name: desired_value}.
        Example: {"earth_grav_field": "EGM96", "Nmax_grav": 10}
    pk_column : str
        Name of the primary key column (default: "id").

    Returns
    -------
    list
        List of primary key values.
    """

    cursor = conn.cursor()

    if filters:
        clauses = []
        values = []
        for col, val in filters.items():
            if val is None:
                clauses.append(f"{col} IS NULL")
            else:
                clauses.append(f"{col} = ?")
                values.append(val)

        where_clause = " AND ".join(clauses)
        query = f"SELECT {pk_column} FROM {table_name} WHERE {where_clause}"
    else:
        query = f"SELECT {pk_column} FROM {table_name}"
        values = []

    cursor.execute(query, values)
    results = [row[0] for row in cursor.fetchall()]

    #conn.close()
    return results

def retrieve_best_rows(window_days, RMS_lim, n_best, config_filters):

    error_RMS_column_name = 'error_RMS_'+str(window_days)+'_day_avg'
    results = retrieve_columns_from_condition('EEI_postproc_results_table',
                                                     'config_table',
                                                     error_RMS_column_name,
                                                     'satellites',
                                                     error_RMS_column_name,
                                                     RMS_lim, 
                                                     'estimation_key',
                                                     '<')
    filtered_keys = get_primary_keys(config_filters)
    results = [res for res in results if (res[0] in filtered_keys)]

    #all_keys = [row[0] for row in results]
    all_rms = [row[1] for row in results]
    #all_constellations = [row[2] for row in results]
    unique_rms, unique_idxs = np.unique(all_rms, return_index=True)
    
    #sorted_idxs = np.argsort(all_rms)
    #n_max = n_best if n_best is not None else len(sorted_idxs)
    n_max = n_best if n_best is not None else len(unique_idxs)

    #sorted_results = [results[i] for i in sorted_idxs[:n_max]]
    sorted_results = [results[i] for i in unique_idxs[:n_max]]

    #best_constellations = [all_constellations[i] for i in sorted_idxs]

    return sorted_results