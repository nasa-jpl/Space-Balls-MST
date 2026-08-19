import sys, os
from SpaceBalls.paths import CONFIG_DIR
import pandas as pd

def parse_goco_file(filepath):
    lines = []

    with open(filepath, 'r') as f:
        in_data = False

        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Detect start of data
            if line.startswith('end_of_head'):
                in_data = True
                continue

            if not in_data:
                continue

            parts = line.split()
            key = parts[0]
            if key=="gfct" or key=="gfc":
                lines.append({'L': parts[1], 'M': parts[2], 'C': parts[3], 'S': parts[4]})

    return pd.DataFrame(lines)

file_name = "GOCO2025s"
coeff_table = parse_goco_file(os.path.join(CONFIG_DIR, 'earth', 'gravity', file_name+".gfc"))

with open(os.path.join(CONFIG_DIR, 'earth', 'gravity', file_name+"_monte_grv.py"), "a") as f:
    # write all J coeffs first
    for _, row in coeff_table.iterrows():
        if int(row['M'])==0 and int(row['L'])>1:
            f.write(f"jCof["+row['L']+"] = "+f"{-float(row.C): .12e}"+";\n")
    f.write("\n")
    # write the rest
    for _, row in coeff_table.iterrows():
        if int(row['M'])>0 and int(row['L'])>1:
            f.write(f"cCof["+row['L']+"]["+row['M']+"] = "+row['C']+";    sCof["+row['L']+"]["+row['M']+"] = "+row['S']+";\n")



print("data parsed")