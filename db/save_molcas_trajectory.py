






def save_molcas_trajectory():









if __name__ == "__main__":
  prefix = '/home/ubuntu/'
  geometry_base_dir = prefix + 'fulvene/geometries/geom_scan_200/'
  calculations_base_dir = prefix + 'fulvene/openmolcas_calculations/geom_scan_200_ANO-L-VTZ/'
  geometry_idxs = np.arange(199)
  n_basis = 36
  db_path = './data/geom_scan_199_molcas_ANO-S-VDZ_2.db'

  save_molcas_calculations_to_db(geometry_base_dir, calculations_base_dir, geometry_idxs, db_path, n_basis=n_basis)