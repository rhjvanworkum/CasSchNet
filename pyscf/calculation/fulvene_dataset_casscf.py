import os
import numpy as np
import time

from pyscf import gto, mcscf, lib, scf

if __name__ == "__main__":
  geometries_base_path = '/home/ubuntu/fulvene/geometries/geom_scan_200/'
  output_path = '/home/ubuntu/fulvene/casscf_calculations/geom_scan_200_sto_6g/'
  n_geometries = 200

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  for i in range(n_geometries):

    # make MOL
    mol = gto.M(atom=geometries_base_path + "geometry_" + str(i) + ".xyz",
                basis="sto_6g",
                spin=0,
                symmetry=True)
    
    # run HF
    myhf = mol.RHF()
    myhf.kernel()
    S = myhf.get_ovlp(mol)

    # initiate CASSCF object
    n_states = 2
    weights = np.ones(n_states) / n_states
    mcas = myhf.CASSCF(ncas=6, nelecas=6).state_average(weights)
    mcas.conv_tol = 1e-8

    # # project initial guess
    # mo = mcscf.project_init_guess(mcas, scf.mo_coeff)
    # mo = mcas.sort_mo([19, 20, 21, 22, 23, 24], mo)

    # guess = scf.hf.init_guess_by_huckel(mol)
    mo = mcscf.project_init_guess(mcas, myhf.mo_coeff)
    mo = mcas.sort_mo([19, 20, 21, 22, 23, 24], mo)

    # run CASSCF
    tstart = time.time()
    (imacro, _, _, _, mo_coeffs, _) = mcas.kernel(mo)
    t_tot = time.time() - tstart
    F = mcas.get_fock()

    print(mcas.mo_occ)

    F = mcas.get_fock(mo_coeff=mo, ci=fcivec)
    mo_e, _ = scipy.linalg.eigh(F, S)
    mo_e = np.abs(mo_energy - mo_e)

    print(imacro)

    assert np.allclose(F, F.T)

    # save CASSCF result
    # np.savez(output_path + 'geometry_' + str(i) + '.npz',
    #          t_tot=t_tot,
    #          imacro=imacro,
    #          mo_coeffs=mo_coeffs,
    #          guess=mo,
    #          S=S,
    #          F=F)

    print(i / n_geometries * 100, ' %')
