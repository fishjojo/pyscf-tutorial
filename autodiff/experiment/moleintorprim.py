from jax import core
from jax import lax
from jax import abstract_arrays
from jax.interpreters import batching
from jax.interpreters import ad
import jax.numpy as jnp
import numpy as np

int1e_ovlp_p = core.Primitive('int1e_ovlp')

def int1e_ovlp(mol):
    return int1e_ovlp_p.bind(mol)

def int1e_ovlp_impl(mol):
    return mol.mol.intor("int1e_ovlp")

def int1e_ovlp_abstract_eval(mol):
    nao = mol.mol.nao
    res_dtype = np.float64
    res_shape = [nao,]*2
    return abstract_arrays.ShapedArray(res_shape, res_dtype)

def int1e_ovlp_jvp_rule(arg_values, arg_tangents):
    mol, = arg_values
    def make_zero(tan):
        return lax.zeros([mol.mol.natm,3]) if type(tan) is ad.Zero else tan

    mol_t, = arg_tangents
    coords = mol_t.coords
    primal_out = int1e_ovlp_p.bind(mol)

    nao = mol.mol.nao
    atmlst = range(mol.mol.natm)
    aoslices = mol.mol.aoslice_by_atom()
    tangent_out = jnp.zeros((nao,nao))

    s1 = -mol.mol.intor("int1e_ipovlp", comp=3)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        tmp = jnp.einsum('xij,x->ij',s1[:,p0:p1],coords[k])
        tangent_out = jax.ops.index_add(tangent_out, jax.ops.index[p0:p1], tmp)
        tangent_out = jax.ops.index_add(tangent_out, jax.ops.index[:,p0:p1], tmp.T)
    return primal_out, tangent_out


int1e_ovlp_p.def_impl(int1e_ovlp_impl)
int1e_ovlp_p.def_abstract_eval(int1e_ovlp_abstract_eval)
ad.primitive_jvps[int1e_ovlp_p] = int1e_ovlp_jvp_rule
