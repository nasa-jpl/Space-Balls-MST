import sqlite3
import sys, os
from SpaceBalls.utils import load_input_file, make_dict_hash_key
from SpaceBalls.paths import INPUT_DIR, CONFIG_DIR
sys.path.insert(0, str(CONFIG_DIR.parent))  # parent of 'config'
import config.constants as constants
Re = constants.earth_radius()

conn = sqlite3.connect(os.path.join(INPUT_DIR,'inputs_database.db'))
c = conn.cursor()

def initialize_db():

    conn.execute("PRAGMA journal_mode=WAL;")

    c.execute("""CREATE TABLE IF NOT EXISTS orbits_table (
              orbit_name TEXT PRIMARY KEY, 
              a_0 NUMERIC NOT NULL, 
              e_0 NUMERIC NOT NULL, 
              i_0 NUMERIC NOT NULL, 
              RAAN_0 NUMERIC NOT NULL, 
              w_0 NUMERIC NOT NULL, 
              theta_0 NUMERIC NOT NULL,
              delta_jd_0 NUMERIC NOT NULL)""") # uniqueness of rows not enfored because some orbits (e.g. A1 and C1 are duplicated for legacy reasons)

    c.execute("""CREATE TABLE IF NOT EXISTS sc_table (
                sc_name TEXT PRIMARY KEY,
                mass NUMERIC,
                area NUMERIC,
                shape TEXT,
                CD NUMERIC,
                diffReflect NUMERIC,
                diffDegrade NUMERIC,
                specReflect NUMERIC,
                specDegrade NUMERIC,
                UNIQUE (mass, area, shape, CD, diffReflect, diffDegrade, specReflect, specDegrade)
                )""")
    
    c.execute("""CREATE TABLE IF NOT EXISTS force_settings_table (
              settings_num INTEGER PRIMARY KEY,
              earth_grav_field TEXT,
              Nmax_grav INTEGER,
              third_bodies TEXT,
              EEI_truth INTEGER,
              smooth_shadow INTEGER,
              custom_accels_key TEXT,
              UNIQUE (earth_grav_field, Nmax_grav, third_bodies, EEI_truth, smooth_shadow, custom_accels_key),
              FOREIGN KEY(custom_accels_key) REFERENCES custom_acc_table(custom_accels_key)
              )""")
    
    c.execute(""" CREATE TABLE IF NOT EXISTS integration_settings_table (
              settings_num INTEGER PRIMARY KEY,
              n_rings INTEGER
              )""") # removed UNIQUE (n_rings)
    
    c.execute("""CREATE TABLE IF NOT EXISTS custom_acc_table (
                custom_accels_key TEXT PRIMARY KEY
                )""")
    
    c.execute("""CREATE TABLE IF NOT EXISTS sb_table (
              sb_hash TEXT PRIMARY KEY,
              orbit_name TEXT NOT NULL,
              sc_name TEXT NOT NULL,
              force_settings INTEGER NOT NULL,
              integration_settings INTEGER NOT NULL,
              delta_t_out NUMERIC NOT NULL,
              FOREIGN KEY(orbit_name) REFERENCES orbits_table(orbit_name),
              FOREIGN KEY(sc_name) REFERENCES   sc_table(sc_name),
              FOREIGN KEY(force_settings) REFERENCES   force_settings_table(settings_num),
              FOREIGN KEY(integration_settings) REFERENCES  integration_settings_table(settings_num)
              );""")    # uniqueness is enforced by the hash of the primary key!



initialize_db()

def insert_sb(orbit_name, sc_name, force_settings_num, integration_settings_num, delta_t_out):
    sb_dict = {
        'orbit_name': orbit_name,
        'sc_name': sc_name,
        'force_settings': force_settings_num,
        'integration_settings': integration_settings_num,
        'delta_t_out': delta_t_out
    }
    sb_hash = make_dict_hash_key(sb_dict)
    sb_dict['hash'] = sb_hash
    with conn:
        c.execute(#"""INSERT INTO sb_table (sb_hash, orbit_name, sc_name, force_settings, integration_settings, delta_t_out)
                  """INSERT OR IGNORE INTO sb_table (sb_hash, orbit_name, sc_name, force_settings, integration_settings, delta_t_out)
                  VALUES (:hash, :orbit_name, :sc_name, :force_settings, :integration_settings, :delta_t_out);
                  """, sb_dict )
    return sb_hash
        
def insert_sb_with_name(sb_name, orbit_name, sc_name, force_settings_num, integration_settings_num, delta_t_out):
    sb_dict = {
        'hash': sb_name,
        'orbit_name': orbit_name,
        'sc_name': sc_name,
        'force_settings': force_settings_num,
        'integration_settings': integration_settings_num,
        'delta_t_out': delta_t_out
    }
    with conn:
        c.execute("""INSERT INTO sb_table (sb_hash, orbit_name, sc_name, force_settings, integration_settings, delta_t_out)
                  VALUES (:hash, :orbit_name, :sc_name, :force_settings, :integration_settings, :delta_t_out);
                  """, sb_dict )


def insert_integration_settings(settings_num, n_rings):
    with conn:
        c.execute("""INSERT INTO integration_settings_table (settings_num, n_rings)
                  VALUES (:num, :n_rings)
                  """, # ON CONFLICT (n_rings) DO NOTHING; - removed
                  {'num': settings_num, 'n_rings': n_rings})

def insert_force_settings(settings_num, earth_field, Nmax_grav, third_bodies, EEI_truth, smooth_shadow_num, custom_accels_key):
    with conn:
        c.execute("""INSERT INTO force_settings_table 
                  (settings_num, earth_grav_field, Nmax_grav, third_bodies, EEI_truth, smooth_shadow, custom_accels_key) 
                  VALUES (:num, :earth_field, :Nmax_grav, :third_bodies, :EEI_truth, :smooth, :custom_accels_key)
                  ON CONFLICT DO NOTHING;""",
                  {'num': settings_num, 'earth_field': earth_field, 'Nmax_grav': Nmax_grav, 'third_bodies': third_bodies,
                   'EEI_truth': EEI_truth, 'smooth': smooth_shadow_num, 'custom_accels_key': custom_accels_key})
                # ON CONFLICT (earth_grav_field, Nmax_grav, third_bodies, EEI_truth, smooth_shadow, custom_accels_key) DO NOTHING;""",

def insert_orbit(sc_tag, kep_0_dict, delta_jd_0=0):
    with conn:
        h = kep_0_dict['a'] - 6378.14 # hard-coded radius used in legacy input files
        a = Re + h
        c.execute("""INSERT INTO orbits_table (orbit_name, a_0, e_0, i_0, RAAN_0, w_0, theta_0, delta_jd_0)
                   VALUES (:name,:a, :e, :i, :o, :u, :t, :delta_jd_0)
                  ON CONFLICT(orbit_name) DO NOTHING;""", 
                  {'name': sc_tag, 'a': a, 'e': kep_0_dict['e'], 'i': kep_0_dict['i'], 
                   'o': kep_0_dict['o'], 'u': kep_0_dict['u'], 't': kep_0_dict['theta'], 'delta_jd_0': delta_jd_0})


def insert_sc_entry(sc_name, sc_input_dict):
    sc_input_dict['name'] = sc_name
    with conn:
        c.execute("""INSERT INTO sc_table (sc_name, mass, area, shape, CD, diffReflect, diffDegrade, specReflect, specDegrade)
                VALUES (:name, :mass, :area, :shape, :cd, :diffReflect, :diffDegrade, :specReflect, :specDegrade)
                    ON CONFLICT (mass, area, shape, CD, diffReflect, diffDegrade, specReflect, specDegrade) DO NOTHING""", 
                    sc_input_dict)


# def insert_config_entry(config_dict: dict, config_dict_hash_key: str):
#    with conn:
#         c.execute("""INSERT INTO config_table (estimation_key, EEI_truth_idx, satellites) VALUES 
#                   (:estimation_key, :EEI_truth_idx, :satellites)
#                   ON CONFLICT(estimation_key) DO NOTHING;""", # , multi-entry insertion
#                   {'estimation_key': config_dict_hash_key, 
#                    'EEI_truth_idx': int(''.join(filter(str.isdigit, config_dict["EEI_truth_name"]))), # config_dict["EEI_truth_name"], 
#                    'satellites': make_list_str_key(config_dict["satellite_list"])}
#                 )
# 

def fetch_one_as_dict(cur, table, key_name, row_id):
    cur.execute(f"SELECT * FROM {table} WHERE {key_name} = ?", (row_id,))
    row = cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def get_dicts(main_id):
    cur = conn.cursor()

    # 1. Get foreign keys from main table
    cur.execute("""
        SELECT orbit_name, sc_name, force_settings, integration_settings, delta_t_out
        FROM sb_table
        WHERE sb_hash = ?
    """, (main_id,))
    
    row = cur.fetchone()
    if row is None:
        return None

    orbit_name, sc_name, force_settings, integration_settings, delta_t_out = row

    result = {}

    # 2. Fetch table rows
    result["orbit"] = fetch_one_as_dict(cur, "orbits_table", "orbit_name", orbit_name)
    result["sc_params"] = fetch_one_as_dict(cur, "sc_table", "sc_name", sc_name)
    result["force_settings"] = fetch_one_as_dict(cur, "force_settings_table", "settings_num", force_settings)
    result["integration_settings"] = fetch_one_as_dict(cur, "integration_settings_table", "settings_num", integration_settings)
    result["delta_t_out"] = delta_t_out


    return result



def get_primary_keys(filters, table_name="sb_table", pk_column="sb_hash"):
    

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

    #conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Build WHERE clause dynamically
    if filters:
        where_clause = " AND ".join([f"{col} = ?" for col in filters.keys()])
        values = list(filters.values())
        query = f"SELECT {pk_column} FROM {table_name} WHERE {where_clause}"
    else:
        # No filters → return all PKs
        query = f"SELECT {pk_column} FROM {table_name}"
        values = []

    cursor.execute(query, values)

    # Extract results
    results = [row[0] for row in cursor.fetchall()]

    #conn.close()
    return results