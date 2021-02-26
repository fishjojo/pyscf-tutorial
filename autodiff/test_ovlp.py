import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
import numpy as np

def grad_analyt(mol):
    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    nao = mol.nao
    g = np.zeros((nao,nao,mol.natm,3))
    s1 = -mol.intor('int1e_ipovlp', comp=3)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        g[p0:p1,:,k] += s1[:,p0:p1].transpose(1,2,0)
        g[:,p0:p1,k] += s1[:,p0:p1].transpose(2,1,0)
    return g

if __name__ == "__main__":
    import pyscf
    from pyscf.gto import mole
    from ad import mole as admole

    mol = pyscf.M(
        atom = 'O 0. 0. 0.; H 0. , -0.757 , 0.587; H 0. , 0.757 , 0.587',
        basis = 'sto3g',
        verbose=0,
    )
    s0 = mol.intor('int1e_ovlp')

    x = mol.atom_coords()
    mol1 = admole.Mole(mol, coords=x)

    def func(mol1):
        return jnp.linalg.norm(mol1.intor("int1e_ovlp"))

    g0 = np.einsum("ij,ijnx->nx", s0, grad_analyt(mol)) / np.linalg.norm(s0)

    jac = jax.jacfwd(func)(mol1)
    g = jac.coords
    print(abs(g-g0).max())
