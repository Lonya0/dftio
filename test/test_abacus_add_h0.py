import os
import shutil
from dftio.io.abacus.abacus_parser import AbacusParser
from tqdm import tqdm


def main():
    parser = AbacusParser(
        root=r'/share/mp_20_abacus_production/15_elements_0524/train_50_nscf/cooking',
        prefix=r'*/cooking/db_seq_id_*',
    )
    num_entries = len(parser.raw_datas)
    outroot = r'/share/mp_20_abacus_production/15_elements_0524/50_H0_workspace_v2/raw_lmdb_data'

    if os.path.exists(outroot):
        import shutil
        shutil.rmtree(outroot)

    raw_outroot = os.path.join(outroot, 'raw')
    old_lmdb_path = os.path.join(raw_outroot, "data.{}.lmdb".format(os.getpid()))
    new_lmdb_path = os.path.join(outroot, 'h0', "data.h0.lmdb")
    for idx in tqdm(range(num_entries)):
        parser.write(
            idx=int(idx),
            format='lmdb',
            hamiltonian=True,
            overlap=True,
            outroot=raw_outroot,
            eigenvalue=True,
            density_matrix=True,
            band_index_min=0
        )
    parser.add_h0_delta_h(
        h0_src_root= r'/share/mp_20_abacus_production/15_elements_0524/50_H0_workspace_v2/cooking',
        old_lmdb_path=old_lmdb_path,
        new_lmdb_path=new_lmdb_path,
        keep_old_lmdb=True
    )


if __name__ == "__main__":
    main()
