import numpy
import pyscf
from pyscf import scf, ao2mo
from pyscf.scf import hf
import jax.ops
from jax import grad, custom_jvp
import jax.numpy as np
import jax.scipy as scipy
from jax.scipy.optimize import minimize
from jax.config import config
config.update("jax_enable_x64", True)
from pyscf import grad as pyscf_grad

mol = pyscf.M(
    atom = 'H 0 0 0; H 0 0 1.1',  # in Angstrom
    basis = '631g',
    verbose=0,
)
coords0 = np.asarray(mol.atom_coords())

def get_init_guess(mol):
    mf = scf.RHF(mol)
    dm0 = mf.get_init_guess()

    h1e = mf.get_hcore(mol)
    s1e = mf.get_ovlp(mol)
    vhf = mf.get_veff(mol, dm0)
    fock = mf.get_fock(h1e, s1e, vhf, dm0)
    mo_energy, mo_coeff = mf.eig(fock, s1e)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    if mf._eri is not None:
        mf._eri = np.asarray(ao2mo.restore(1, mf._eri, mol.nao))
    return mf, mo_coeff, mo_occ, dm0

"""
mf, mo_coeff, mo_occ, dm0 = get_init_guess(mol)
'''
dm = mf.make_rdm1(mo_coeff, mo_occ)
print(mf.energy_elec(dm,h1e,vhf))
dx = mf.get_grad(mo_coeff,mo_occ)
R = hf.unpack_uniq_var(dx, mo_occ)
print(R)
'''
e_ref = mf.kernel()
print(e_ref)

nao = mol.nao
size = nao*(nao+1)//2
x0 = np.zeros([size,])
"""

def inter_distance(mol, coords):
    coords = np.asarray(coords).reshape(-1,3)
    rr = np.linalg.norm(coords.reshape(-1,1,3) - coords, axis=2)
    rr = jax.ops.index_update(rr, np.diag_indices_from(rr), 0.)
    return rr

def energy_nuc(mol, coords, charges=None):
    coords = np.asarray(coords).reshape(-1,3)
    if charges is None: charges = np.asarray(mol.atom_charges())
    rr = inter_distance(mol, coords)
    rr = jax.ops.index_update(rr, np.diag_indices_from(rr), 1.e200)
    e = np.einsum('i,ij,j->', charges, 1./rr, charges) * .5
    return e

def unpack_triu(x, n, hermi=0):
    R = np.zeros([n,n])
    idx = np.triu_indices(n)
    R = jax.ops.index_update(R, idx, x)
    if hermi == 0:
        return R
    elif hermi == 1:
        R = R + R.conj().T
        R = jax.ops.index_mul(R, np.diag_indices(n), 0.5) 
        return R
    elif hermi == 2:
        return R - R.conj().T
    else:
        raise KeyError

def update_rotate_matrix(dx, n, u0=1):
    dr = unpack_triu(dx, n, hermi=2)
    u = np.dot(u0, scipy.linalg.expm(dr))
    return u

def rotate_mo(mo_coeff, u):
    mo = np.dot(mo_coeff, u)
    return mo

def get_jk(mf, dm, with_j=True, with_k=True):
    mol = mf.mol
    nao = dm.shape[0]
    if mf._eri is None:
        mf._eri = np.asarray(mol.intor('int2e', aosym='s1')).reshape([nao,]*4)
    eri = mf._eri

    if with_j:
        vj = np.einsum('ijkl,ji->kl', eri, dm)
    if with_k:
        vk = np.einsum('ijkl,jk->il', eri, dm)
    return vj, vk

def get_veff(mf, dm):
    vj, vk = get_jk(mf, dm)
    return vj - 0.5*vk

def energy_elec(mf, dm):
    h1e = mf.get_hcore()
    vhf = get_veff(mf, dm)
    e1 = np.einsum('ij,ji->',  h1e, dm)
    e_coul = np.einsum('ij,ji->', vhf, dm) * .5
    return e1 + e_coul

def make_rdm1(mo_coeff, mo_occ, **kwargs):
    mocc = mo_coeff[:,mo_occ>0]
    return np.dot(mocc*mo_occ[mo_occ>0], mocc.conj().T)

def energy_tot(x, coords, mf, mo_coeff, mo_occ):
    mol = mf.mol
    nao = mo_coeff.shape[0]
    u = update_rotate_matrix(x, nao)
    mo_coeff = rotate_mo(mo_coeff, u)
    dm = make_rdm1(mo_coeff, mo_occ)
    e_tot = energy_elec(mf, dm) + energy_nuc(mol,coords)
    return e_tot

def jac(x, mf, mo_coeff, mo_occ):
    g = grad(func,0)(x, mf, mo_coeff, mo_occ)
    return g

def run_scf(x0, coords, mf, mo_coeff, mo_occ):
    options = {"gtol":1e-6}
    res = minimize(energy_tot, x0, args=(coords, mf, mo_coeff, mo_occ), method="BFGS", options = options)
    e = energy_tot(res.x, coords, mf, mo_coeff, mo_occ)
    print("SCF energy: ", e)


def scanner(coords, mol):
    coords = numpy.asarray(coords).reshape(-1,3)
    mol.set_geom_(coords, unit='B')
    mf, mo_coeff, mo_occ, dm0 = get_init_guess(mol)
    nao = mol.nao
    size = nao*(nao+1)//2
    x0 = np.zeros([size,])
    e = run_scf(x0, coords, mf, mo_coeff, mo_occ)
    return e


#scanner(coords0, mol)
g = grad(scanner, 0)
g(coords0, mol)








