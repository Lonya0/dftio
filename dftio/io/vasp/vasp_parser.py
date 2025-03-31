from scipy.linalg import block_diag
import re
from tqdm import tqdm
from collections import Counter
from dftio.constants import orbitalId
import ase
from ase.io import read, write
import dpdata
import os
import numpy as np
from dftio.io.parse import Parser, ParserRegister, find_target_line
from dftio.data import _keys
from dftio.register import Register

@ParserRegister.register("vasp")
class VASPParser(Parser):
    def __init__(
            self,
            root,
            prefix,
            **kwargs
            ):
        super(VASPParser, self).__init__(root, prefix)
        self.raw_sys = [read(self.raw_datas[idx]+'/POSCAR') for idx in range(len(self.raw_datas))]
    
    # essential
    def get_structure(self, idx):
        sys = self.raw_sys[idx]
        
        structure = self.ase_to_structure(sys)

        return structure
    
    
    # essential
    def get_eigenvalue(self, idx):
        path = self.raw_datas[idx]
        assert os.path.exists(os.path.join(path, "EIGENVAL"))
        kpts, eigs = self.read_EIGENVAL(os.path.join(path, "EIGENVAL"))

        return {_keys.ENERGY_EIGENVALUE_KEY: eigs, _keys.KPOINT_KEY: kpts}
    
    @staticmethod
    def read_EIGENVAL(file):
        Nhse = 0 # number of HSE bands, used for HSE, 
        f = open(file, 'r')
        data = f.readlines()
        # Read the number of bands
        NBND = int(re.findall('[0-9]+', data[5])[2])
        k_list = []
        k_bands = []
        kb_temp = []
        kb_count = 0
        for i in range(7+(NBND+2)*Nhse, len(data)):
            temp = re.findall('[0-9\-\.\+E]+', data[i])
            if not temp:
                continue
            if len(temp) == 4:
                kt = (np.array([float(i) for i in temp[0:3]])).tolist()
                k_list.append(kt)
            else:
                kb_temp.append(float(temp[1])) # the energy.
                kb_count += 1
                if kb_count == NBND:
                    k_bands.append(sorted(kb_temp))
                    kb_temp = []
                    kb_count = 0
        k_bands = np.array(k_bands)[np.newaxis, :, :] # [1, nk, nbands]
        k_list = np.asarray(k_list) # [nk, 3] 

        return k_list, k_bands

 
    def get_blocks(self, idx, hamiltonian: bool=False, overlap: bool=False, density_matrix: bool=False):
        raise NotImplementedError("VASP does not support block parsing yet.")

