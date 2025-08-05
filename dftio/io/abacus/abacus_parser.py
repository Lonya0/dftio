from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
import re
from tqdm import tqdm
from collections import Counter
from dftio.constants import orbitalId, ABACUS2DFTIO
import ase
from ase.io import read
import dpdata
import os
import numpy as np
from dftio.io.parse import Parser, ParserRegister, find_target_line
from dftio.data import _keys
from dftio.register import Register
import lmdb
import pickle
import shutil


@ParserRegister.register("abacus")
class AbacusParser(Parser):
    def __init__(
            self,
            root,
            prefix,
            **kwargs
            ):
        super(AbacusParser, self).__init__(root, prefix)
        mode = self.get_mode(idx=0)
        if mode in ['nscf', "scf"]:
            self.raw_sys = [dpdata.System(read(os.path.join(self.raw_datas[idx], "OUT.ABACUS", "STRU.cif")), fmt="ase/structure") for idx in range(len(self.raw_datas))]
        else:
            self.raw_sys = [dpdata.LabeledSystem(self.raw_datas[idx], fmt='abacus/'+self.get_mode(idx)) for idx in range(len(self.raw_datas))]

    # essential
    def get_structure(self, idx):
        sys = self.raw_sys[idx]
        
        structure = {
            _keys.ATOMIC_NUMBERS_KEY: np.array([ase.atom.atomic_numbers[i] for i in sys.data["atom_names"]], dtype=np.int32)[sys.data["atom_types"]],
            _keys.PBC_KEY: np.array([True, True, True]) # abacus does not allow non-pbc structure
        }
        structure[_keys.POSITIONS_KEY] = sys.data["coords"].astype(np.float32)
        structure[_keys.CELL_KEY] = sys.data["cells"].astype(np.float32)

        return structure
    
    def get_mode(self, idx):
        with open(os.path.join(self.raw_datas[idx], "OUT.ABACUS", "INPUT"), 'r') as f:
            line = find_target_line(f, "calculation")
            assert line is not None, 'Cannot find "MODE" in log file'
            mode = line.split()[1]
            f.close()

        return mode
    
    # essential
    def get_eigenvalue(self, idx, band_index_min=0):
        path = self.raw_datas[idx]
        mode = self.get_mode(idx)
        if mode in ["scf", "nscf"]:
            assert os.path.exists(os.path.join(path, "OUT.ABACUS", "BANDS_1.dat"))
            eigs = np.loadtxt(os.path.join(path, "OUT.ABACUS", "BANDS_1.dat"))[np.newaxis, :, 2+band_index_min:]
            assert os.path.exists(os.path.join(path, "OUT.ABACUS", "kpoints"))
            kpts = []
            with open(os.path.join(path, "OUT.ABACUS", "kpoints"), "r") as f:
                line = find_target_line(f, "nkstot now")
                nkstot = line.strip().split()[-1]
                line = find_target_line(f, "KPOINTS ")
                if not line:
                    f.seek(0)
                    line = find_target_line(f, "KPT ")
                if not line: 
                    raise Exception("Cannot find KPT or KPOINTS in kpoints file")
                for _ in range(int(nkstot)):
                    line = f.readline()
                    kpt = []
                    line = line.strip().split()
                    kpt.extend([float(line[1]), float(line[2]), float(line[3])])
                    kpts.append(kpt)
                kpts = np.array(kpts)
        elif mode == "md" or mode == "relax":
            raise NotImplementedError("output eigenvalues from MD trajectory is not supported by ABACUS.")
        
        else:
            raise NotImplementedError("mode {} is not supported.".format(mode))
            
        return {_keys.ENERGY_EIGENVALUE_KEY: eigs.astype(np.float32), _keys.KPOINT_KEY: kpts.astype(np.float32)}
    
    # essential
    def get_basis(self, idx):
        # {"Si": "2s2p1d"}
        mode = self.get_mode(idx)
        logfile = "running_"+mode+".log"
        sys = self.raw_sys[idx]
        with open(os.path.join(self.raw_datas[idx], "OUT.ABACUS", logfile), 'r') as f:
            orbital_types_dict = {}
            for index_type in range(len(sys.data["atom_numbs"])):
                tmp = find_target_line(f, "READING ATOM TYPE")
                assert tmp is not None, 'Cannot find "ATOM TYPE" in log file'
                assert tmp.split()[-1] == str(index_type + 1)
                if tmp is None:
                    raise Exception(f"Cannot find ATOM {index_type} in {logfile}")

                line = f.readline()
                assert "atom label =" in line
                atom_label = line.split()[-1]
                atom_label = ''.join(re.findall(r'[A-Za-z]', atom_label))
                assert atom_label in ase.data.atomic_numbers, "Atom label should be in periodic table"

                current_orbital_types = []
                while True:
                    line = f.readline()
                    if "number of zeta" in line:
                        tmp = line.split()
                        L = int(tmp[0][2:-1])
                        num_L = int(tmp[-1])
                        current_orbital_types.extend([L] * num_L)
                    else:
                        break
                orbital_types_dict[atom_label] = current_orbital_types
        basis = {}
        for k,v in orbital_types_dict.items():
            counter = Counter(v)
            basis[k] = [str(counter[l])+orbitalId[l] for l in range(max(counter.keys())+1) if counter.get(l, 0) > 0]
            basis[k] = "".join(basis[k])
        
        return basis
    
    # essential
    def get_blocks(self, idx, hamiltonian=True, overlap=False, density_matrix=False):
        mode = self.get_mode(idx)
        logfile = "running_"+mode+".log"
        hamiltonian_dict, overlap_dict, density_matrix_dict = None, None, None
        sys = self.raw_sys[idx]
        nsites = sys.data["atom_types"].shape[0]
        if os.path.exists(os.path.join(self.raw_datas[idx], "OUT.ABACUS", "hscsr.tgz")):
            os.system(f"tar -xzf {os.path.join(self.raw_datas[idx], 'OUT.ABACUS', 'hscsr.tgz')} -C {os.path.join(self.raw_datas[idx])}")
        with open(os.path.join(self.raw_datas[idx], "OUT.ABACUS", logfile), 'r') as f:
            site_norbits_dict = {}
            orbital_types_dict = {}
            for index_type in range(len(sys.data["atom_numbs"])):
                tmp = find_target_line(f, "READING ATOM TYPE")
                assert tmp is not None, 'Cannot find "ATOM TYPE" in log file'
                assert tmp.split()[-1] == str(index_type + 1)
                if tmp is None:
                    raise Exception(f"Cannot find ATOM {index_type} in {logfile}")

                line = f.readline()
                assert "atom label =" in line
                atom_label = line.split()[-1]
                atom_label = ''.join(re.findall(r'[A-Za-z]', atom_label))
                assert atom_label in ase.data.atomic_numbers, "Atom label should be in periodic table"
                atom_type = ase.data.atomic_numbers[atom_label]

                current_site_norbits = 0
                current_orbital_types = []
                while True:
                    line = f.readline()
                    if "number of zeta" in line:
                        tmp = line.split()
                        L = int(tmp[0][2:-1])
                        num_L = int(tmp[-1])
                        current_site_norbits += (2 * L + 1) * num_L
                        current_orbital_types.extend([L] * num_L)
                    else:
                        break
                site_norbits_dict[atom_type] = current_site_norbits
                orbital_types_dict[atom_type] = current_orbital_types


            line = find_target_line(f, " COORDINATES")
            assert "atom" in f.readline()
            site_norbits = np.zeros(nsites, dtype=int)
            element = np.zeros(nsites, dtype=int)
            for index_site in range(nsites):
                line = f.readline()
                tmp = line.split()
                assert "tau" in tmp[0]
                atom_label = ''.join(re.findall(r'[A-Za-z]', tmp[0][5:]))
                assert atom_label in ase.data.atomic_numbers, "Atom label should be in periodic table"
                element[index_site] = ase.data.atomic_numbers[atom_label]
                site_norbits[index_site] = site_norbits_dict[element[index_site]]

            if hamiltonian is False and overlap is True:
                spinful = False
            else:
                line = find_target_line(f, "nspin")
                if line is None:
                    line = find_target_line(f, "NSPIN")
                assert line is not None, 'Cannot find "NSPIN" in log file'
                if "NSPIN == 1" or "npin = 1" in line:
                    spinful = False
                elif "NSPIN == 4" or "nspin = 4" in line:
                    spinful = True
                else:
                    raise ValueError(f'{line} is not supported')

        if mode in ["scf", "nscf"]:
            if hamiltonian:
                hamiltonian_dict, tmp = self.parse_matrix(
                    matrix_path=os.path.join(self.raw_datas[idx], "OUT.ABACUS", "data-HR-sparse_SPIN0.csr"), 
                    nsites=nsites,
                    site_norbits=site_norbits,
                    orbital_types_dict=orbital_types_dict,
                    element=element,
                    factor=13.605698, # Ryd2eV
                    spinful=spinful
                    )
                assert tmp == int(np.sum(site_norbits)) * (1 + spinful)
                hamiltonian_dict = [hamiltonian_dict]
            
            if overlap:
                overlap_dict, tmp = self.parse_matrix(
                    matrix_path=os.path.join(self.raw_datas[idx], "OUT.ABACUS", "data-SR-sparse_SPIN0.csr"), 
                    nsites=nsites,
                    site_norbits=site_norbits,
                    orbital_types_dict=orbital_types_dict,
                    element=element,
                    factor=1,
                    spinful=spinful
                    )
                assert tmp == int(np.sum(site_norbits)) * (1 + spinful)

                if spinful:
                    overlap_dict_spinless = {}
                    for k, v in overlap_dict.items():
                        overlap_dict_spinless[k] = v[:v.shape[0] // 2, :v.shape[1] // 2].real
                    overlap_dict_spinless, overlap_dict = overlap_dict, overlap_dict_spinless

                overlap_dict = [overlap_dict]

            if density_matrix:
                density_matrix_dict, tmp = self.parse_matrix(
                    matrix_path=os.path.join(self.raw_datas[idx], "OUT.ABACUS", "data-DMR-sparse_SPIN0.csr"), 
                    nsites=nsites,
                    site_norbits=site_norbits,
                    orbital_types_dict=orbital_types_dict,
                    element=element,
                    factor=1,
                    spinful=spinful
                    )
                assert tmp == int(np.sum(site_norbits)) * (1 + spinful)

                density_matrix_dict = [density_matrix_dict]

        elif mode == "md":
            if hamiltonian:
                hamiltonian_dict = []
                for i in range(sys.get_nframes()):
                    hamil, tmp = self.parse_matrix(
                        matrix_path=os.path.join(self.raw_datas[idx], "OUT.ABACUS", "matrix/"+str(i)+"_data-HR-sparse_SPIN0.csr"), 
                        nsites=nsites,
                        site_norbits=site_norbits,
                        orbital_types_dict=orbital_types_dict,
                        element=element,
                        factor=13.605698, # Ryd2eV
                        spinful=spinful
                        )
                    assert tmp == int(np.sum(site_norbits)) * (1 + spinful)
                    hamiltonian_dict.append(hamil)

            if overlap:
                overlap_dict = []
                for i in range(sys.get_nframes()):
                    ovp, tmp = self.parse_matrix(
                        matrix_path=os.path.join(self.raw_datas[idx], "OUT.ABACUS", "matrix/"+str(i)+"_data-SR-sparse_SPIN0.csr"), 
                        nsites=nsites,
                        site_norbits=site_norbits,
                        orbital_types_dict=orbital_types_dict,
                        element=element,
                        factor=1,
                        spinful=spinful
                        )
                    assert tmp == int(np.sum(site_norbits)) * (1 + spinful)

                    if spinful:
                        ovp_spinless = {}
                        for k, v in ovp.items():
                            ovp_spinless[k] = v[:v.shape[0] // 2, :v.shape[1] // 2].real

                        ovp_spinless, ovp = ovp, ovp_spinless
                    overlap_dict.append(ovp)
            
            if density_matrix:
                density_matrix_dict = []
                for i in range(sys.get_nframes()):
                    dm, tmp = self.parse_matrix(
                        matrix_path=os.path.join(self.raw_datas[idx], "OUT.ABACUS", "matrix/"+str(i)+"_data-DMR-sparse_SPIN0.csr"), 
                        nsites=nsites,
                        site_norbits=site_norbits,
                        orbital_types_dict=orbital_types_dict,
                        element=element,
                        factor=1,
                        spinful=spinful
                        )
                    assert tmp == int(np.sum(site_norbits)) * (1 + spinful)
                    density_matrix_dict.append(dm)
        else:
            raise NotImplementedError("mode {} is not supported.".format(mode))
        
        return hamiltonian_dict, overlap_dict, density_matrix_dict

    def parse_matrix(self, matrix_path, nsites, site_norbits, orbital_types_dict, element, factor, spinful=False):
        site_norbits_cumsum = np.cumsum(site_norbits)
        norbits = int(np.sum(site_norbits))
        matrix_dict = dict()
        with open(matrix_path, 'r') as f:
            line = f.readline() # read "Matrix Dimension of ..."
            if not "Matrix Dimension of" in line:
                line = f.readline() # ABACUS >= 3.0
                assert "Matrix Dimension of" in line
            f.readline() # read "Matrix number of ..."
            norbits = int(line.split()[-1])
            for line in f:
                line1 = line.split()
                if len(line1) == 0 or len(line1) == 2:
                    break
                num_element = int(line1[3])
                if num_element != 0:
                    R_cur = np.array(line1[:3]).astype(int)
                    line2 = f.readline().split()
                    line3 = f.readline().split()
                    line4 = f.readline().split()
                    if not spinful:
                        hamiltonian_cur = csr_matrix((np.array(line2).astype(np.float32), np.array(line3).astype(int),
                                                        np.array(line4).astype(np.int32)), shape=(norbits, norbits), dtype=np.float32).toarray()
                    else:
                        line2 = np.char.replace(line2, '(', '')
                        line2 = np.char.replace(line2, ')', 'j')
                        line2 = np.char.replace(line2, ',', '+')
                        line2 = np.char.replace(line2, '+-', '-')
                        hamiltonian_cur = csr_matrix((np.array(line2).astype(np.complex64), np.array(line3).astype(int),
                                                    np.array(line4).astype(np.int32)), shape=(norbits, norbits), dtype=np.complex64).toarray()
                    for index_site_i in range(nsites):
                        for index_site_j in range(nsites):
                            key_str = f"{index_site_i}_{index_site_j}_{R_cur[0]}_{R_cur[1]}_{R_cur[2]}"
                            mat = hamiltonian_cur[(site_norbits_cumsum[index_site_i]
                                                    - site_norbits[index_site_i]) * (1 + spinful):
                                                    site_norbits_cumsum[index_site_i] * (1 + spinful),
                                    (site_norbits_cumsum[index_site_j] - site_norbits[index_site_j]) * (1 + spinful):
                                    site_norbits_cumsum[index_site_j] * (1 + spinful)]
                            if abs(mat).max() < 1e-10:
                                continue
                            if not spinful:
                                mat = self.transform(mat, orbital_types_dict[element[index_site_i]],
                                                            orbital_types_dict[element[index_site_j]])
                            else:
                                mat = mat.reshape((site_norbits[index_site_i], 2, site_norbits[index_site_j], 2))
                                mat = mat.transpose((1, 0, 3, 2)).reshape((2 * site_norbits[index_site_i],
                                                                        2 * site_norbits[index_site_j]))
                                mat = self.transform(mat, orbital_types_dict[element[index_site_i]] * 2,
                                                        orbital_types_dict[element[index_site_j]] * 2)
                            matrix_dict[key_str] = mat * factor
        return matrix_dict, norbits
    
    def transform(self, mat, l_lefts, l_rights):

        if max(*l_lefts, *l_rights) > 5:
            raise NotImplementedError("Only support l = s, p, d, f, g, h.")
        block_lefts = block_diag(*[ABACUS2DFTIO[l_left] for l_left in l_lefts])
        block_rights = block_diag(*[ABACUS2DFTIO[l_right] for l_right in l_rights])

        return block_lefts @ mat @ block_rights.T

    def get_abs_h0_folders(self, h0_root):
        # Build a map of all directory names to their full paths to avoid repeated os.walk calls
        folder_path_map = {}
        for sub_root, dirs, _ in os.walk(h0_root):
            for dirname in dirs:
                folder_path_map[dirname] = os.path.join(sub_root, dirname)

        abs_h0_folders = []
        valid_idx_list = []
        for valid_idx, a_H_folder in enumerate(self.raw_datas):
            a_leaf_folder_name = os.path.split(a_H_folder)[-1]

            # Use pre-built map for O(1) lookup instead of calling slow find_leaf_folder()
            a_leaf_folder_path = folder_path_map.get(a_leaf_folder_name)
            found_flag = a_leaf_folder_path is not None

            if found_flag:
                abs_h0_folders.append(a_leaf_folder_path)
                valid_idx_list.append(valid_idx)
            else:
                abs_h0_folders.append(None)
                valid_idx_list.append(None)
        return abs_h0_folders, valid_idx_list

    def add_h0_delta_h(self, h0_src_root, old_lmdb_path, new_lmdb_path, keep_old_lmdb: bool = True,
                       keep_delta_ham_only: bool = True):
        h0_root = os.path.abspath(h0_src_root)
        self.raw_datas, valid_idx_list = self.get_abs_h0_folders(h0_root=h0_root)

        os.makedirs(new_lmdb_path, exist_ok=True)
        counter = 0
        batch_size = 50

        # Process in batches
        for batch_start in tqdm(range(0, len(valid_idx_list), batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(valid_idx_list))
            batch_indices = valid_idx_list[batch_start:batch_end]

            # Open LMDB environments for each batch
            old_db_env = lmdb.open(old_lmdb_path, readonly=True, lock=False)
            new_db_env = lmdb.open(new_lmdb_path, map_size=1048576000000, lock=True)

            with old_db_env.begin() as old_txn, new_db_env.begin(write=True) as new_txn:
                for idx in batch_indices:
                    if idx == None:
                        continue
                    # Get data from old LMDB
                    data_dict = old_txn.get(idx.to_bytes(length=4, byteorder='big'))
                    data_dict = pickle.loads(data_dict)

                    # Get H0 block
                    h0_block, _0, _ = self.get_blocks(idx, hamiltonian=True, overlap=False, density_matrix=False)
                    h0_block = h0_block[0]
                    old_ham_block = data_dict['hamiltonian']

                    # Calculate delta blocks
                    delta_block = dict()
                    for a_block_name in old_ham_block.keys():
                        a_delta_block = old_ham_block[a_block_name] - h0_block[a_block_name]
                        delta_block[a_block_name] = a_delta_block

                    # Update data dictionary
                    data_dict['hamiltonian'] = delta_block
                    if not keep_delta_ham_only:
                        data_dict['hamiltonian_full'] = old_ham_block
                        data_dict['hamiltonian_0'] = h0_block

                    # Store in new LMDB
                    data_dict = pickle.dumps(data_dict)
                    new_txn.put(counter.to_bytes(length=4, byteorder='big'), data_dict)
                    counter = counter + 1

            # Close LMDB environments after each batch
            old_db_env.close()
            new_db_env.close()

        # Remove old LMDB if requested
        if not keep_old_lmdb:
            shutil.rmtree(old_lmdb_path)