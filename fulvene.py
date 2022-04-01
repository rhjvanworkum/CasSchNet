from src.schnetpack.data.parser import parse_molcas_rasscf_calculation
from src.schnetpack.data.atoms import AtomsDataFormat, create_dataset, load_dataset

""" generate ASE database """
# calc_dir = 'C:/users/rhjva/imperial/molcas_files/fulvene/'
# db_path = './test.db'

# parse_molcas_rasscf_calculation(calc_dir, db_path)

# from ase.db import connect
# with connect('./test.db') as conn:
#   conn.metadata = {"_distance_unit": 'nm',
#                    "_property_unit_dict": {"orbital_coeffs": 1.0}}

""" Initializing a dataset """
ds = load_dataset(datapath='./test.db', 
                    format=AtomsDataFormat.ASE)