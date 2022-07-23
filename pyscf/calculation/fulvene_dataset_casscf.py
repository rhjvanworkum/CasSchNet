import os
import numpy as np
import time

from pyscf import gto, mcscf, lib

if __name__ == "__main__":
  geometries_base_path = '/home/ubuntu/fulvene/geometries/geom_scan_200/'
  output_path = '/home/ubuntu/fulvene/casscf_calculations/geom_scan_200_sto_6g/'
  n_geometries = 200

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  for i in range(n_geometries):

    # make MOL
    mol = gto.M(atom=geometries_base_path + "geometry_" + str(i) + ".xyz",
                basis="6-31G*",
                spin=0)
    
    # run HF
    scf = mol.RHF()
    scf.kernel()
    S = scf.get_ovlp(mol)

    # initiate CASSCF object
    n_states = 2
    weights = np.ones(n_states) / n_states
    mcas = scf.CASSCF(ncas=6, nelecas=6).state_average(weights)
    mcas.conv_tol = 1e-8

    # project initial guess
    mo = mcscf.project_init_guess(mcas, scf.mo_coeff)
    mo = mcas.sort_mo([19, 20, 21, 22, 23, 24], mo)

    # run CASSCF
    tstart = time.time()
    (imacro, _, _, _, mo_coeffs, _) = mcas.kernel(mo)
    t_tot = time.time() - tstart
    F = mcas.get_fock()

    assert np.allclose(F, F.T)

    # save CASSCF result
    np.savez(output_path + 'geometry_' + str(i) + '.npz',
             t_tot=t_tot,
             imacro=imacro,
             mo_coeffs=mo_coeffs,
             guess=mo,
             S=S,
             F=F)

    print(i / n_geometries * 100, ' %')
