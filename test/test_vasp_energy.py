import os
import pytest
import numpy as np
from dftio.io.vasp.vasp_parser import VASPParser
from dftio.data import _keys


def test_vasp_scf_energy():
    """Test energy extraction for VASP SCF calculation."""
    # Use the test data
    test_data_dir = os.path.join(os.path.dirname(__file__), "data", "vasp_scf")

    parser = VASPParser(
        root=[test_data_dir],
        prefix=""
    )

    # Test get_total_energy method
    energy_data = parser.get_total_energy(idx=0)

    assert energy_data is not None, "Energy data should not be None"
    assert _keys.TOTAL_ENERGY_KEY in energy_data, f"Energy data should contain {_keys.TOTAL_ENERGY_KEY}"

    energy = energy_data[_keys.TOTAL_ENERGY_KEY]

    # Check shape - should be [1,] for SCF
    assert energy.shape == (1,), f"Expected shape (1,), got {energy.shape}"

    # Check dtype
    assert energy.dtype == np.float64, f"Expected dtype float64, got {energy.dtype}"

    # Check value (from OUTCAR: last energy(sigma->0) = -10.53568912 eV)
    expected_energy = -10.53568912
    assert np.isclose(energy[0], expected_energy, atol=1e-5), \
        f"Expected energy ~{expected_energy}, got {energy[0]}"

    print(f"VASP SCF energy extraction test passed: {energy[0]} eV")


def test_vasp_read_total_energy_static():
    """Test the static read_total_energy method directly."""
    test_outcar = os.path.join(os.path.dirname(__file__), "data", "vasp_scf", "OUTCAR")

    energy = VASPParser.read_total_energy(test_outcar)

    # Should extract the last energy(sigma->0) value
    expected_energy = -10.53568912
    assert np.isclose(energy, expected_energy, atol=1e-9, rtol=0), \
        f"Expected energy ~{expected_energy}, got {energy}"

    print(f"VASP read_total_energy static method test passed: {energy} eV")


def test_vasp_energy_write_dat():
    """Test energy writing to dat format."""
    import tempfile

    test_data_dir = os.path.join(os.path.dirname(__file__), "data", "vasp_scf")

    parser = VASPParser(
        root=[test_data_dir],
        prefix=""
    )

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        parser.write(
            idx=0,
            outroot=tmpdir,
            format="dat",
            eigenvalue=False,
            hamiltonian=False,
            overlap=False,
            density_matrix=False,
            band_index_min=0,
            energy=True
        )

        # Check if energy file was created
        output_dir = os.path.join(tmpdir, parser.formula(idx=0) + ".0")
        energy_file = os.path.join(output_dir, "total_energy.dat")

        assert os.path.exists(energy_file), f"Energy file should exist at {energy_file}"

        # Load and verify energy
        loaded_energy = np.loadtxt(energy_file)
        # loadtxt returns scalar for single value, convert to array for consistency
        if loaded_energy.ndim == 0:
            loaded_energy = np.array([loaded_energy])
        assert loaded_energy.shape == (1,), f"Expected shape (1,), got {loaded_energy.shape}"
        assert np.isclose(loaded_energy[0], -10.53568912, atol=1e-5), \
            f"Loaded energy doesn't match expected value"

        print(f"VASP DAT format energy write test passed")


def test_vasp_energy_write_lmdb():
    """Test energy writing to LMDB format."""
    import tempfile
    import lmdb
    import pickle

    test_data_dir = os.path.join(os.path.dirname(__file__), "data", "vasp_scf")

    parser = VASPParser(
        root=[test_data_dir],
        prefix=""
    )

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        parser.write(
            idx=0,
            outroot=tmpdir,
            format="lmdb",
            eigenvalue=False,
            hamiltonian=False,
            overlap=False,
            density_matrix=False,
            band_index_min=0,
            energy=True
        )

        # Check if LMDB was created
        lmdb_path = os.path.join(tmpdir, f"data.{os.getpid()}.lmdb")
        assert os.path.exists(lmdb_path), f"LMDB should exist at {lmdb_path}"

        # Open and verify energy in LMDB
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            data = txn.get((0).to_bytes(length=4, byteorder='big'))
            assert data is not None, "LMDB should contain data"

            data_dict = pickle.loads(data)
            assert _keys.TOTAL_ENERGY_KEY in data_dict, \
                f"LMDB data should contain {_keys.TOTAL_ENERGY_KEY}"

            energy = data_dict[_keys.TOTAL_ENERGY_KEY]
            assert np.isclose(energy, -10.53568912, atol=1e-5), \
                f"LMDB energy doesn't match expected value"

        env.close()
        print(f"VASP LMDB format energy write test passed")


def test_vasp_multiple_energies_warning():
    """Test that warning is logged when multiple energy(sigma->0) found."""
    import tempfile
    import logging
    from io import StringIO

    # Create a temporary OUTCAR with multiple energies
    with tempfile.TemporaryDirectory() as tmpdir:
        test_outcar = os.path.join(tmpdir, "OUTCAR")
        with open(test_outcar, 'w') as f:
            f.write("  energy  without entropy=      -10.84  energy(sigma->0) =      -10.84\n")
            f.write("  energy  without entropy=      -10.54  energy(sigma->0) =      -10.54\n")
            f.write("  energy  without entropy=      -10.53  energy(sigma->0) =      -10.53\n")

        # Capture log output
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger('dftio.io.vasp.vasp_parser')
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

        try:
            energy = VASPParser.read_total_energy(test_outcar)

            # Should return the last value
            assert np.isclose(energy, -10.53, atol=1e-5), \
                f"Expected last energy -10.53, got {energy}"

            # Check if warning was logged
            log_contents = log_stream.getvalue()
            assert "Multiple energy(sigma->0) found" in log_contents, \
                "Warning about multiple energies should be logged"

            print(f"Multiple energies warning test passed")
        finally:
            logger.removeHandler(handler)


def test_vasp_unsuccessful_completion_warning():
    """Test that warning is logged when OUTCAR does not indicate successful completion."""
    import tempfile
    import logging
    from io import StringIO

    # Create a temporary OUTCAR without "Voluntary context switches"
    with tempfile.TemporaryDirectory() as tmpdir:
        test_outcar = os.path.join(tmpdir, "OUTCAR")
        with open(test_outcar, 'w') as f:
            f.write("  energy  without entropy=      -10.53  energy(sigma->0) =      -10.53\n")
            # No "Voluntary context switches" line - simulates unsuccessful completion

        # Capture log output
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger('dftio.io.vasp.vasp_parser')
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

        try:
            energy = VASPParser.read_total_energy(test_outcar)

            # Should still extract energy
            assert np.isclose(energy, -10.53, atol=1e-5), \
                f"Expected energy -10.53, got {energy}"

            # Check if warning was logged about unsuccessful completion
            log_contents = log_stream.getvalue()
            assert "does not indicate successful completion" in log_contents, \
                "Warning about unsuccessful completion should be logged"

            print(f"Unsuccessful completion warning test passed")
        finally:
            logger.removeHandler(handler)


def test_vasp_successful_completion_no_warning():
    """Test that no warning is logged when OUTCAR indicates successful completion."""
    import tempfile
    import logging
    from io import StringIO

    # Create a temporary OUTCAR with "Voluntary context switches"
    with tempfile.TemporaryDirectory() as tmpdir:
        test_outcar = os.path.join(tmpdir, "OUTCAR")
        with open(test_outcar, 'w') as f:
            f.write("  energy  without entropy=      -10.53  energy(sigma->0) =      -10.53\n")
            f.write("    Voluntary context switches: 1234\n")  # Indicates successful completion

        # Capture log output
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger('dftio.io.vasp.vasp_parser')
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

        try:
            energy = VASPParser.read_total_energy(test_outcar)

            # Should extract energy
            assert np.isclose(energy, -10.53, atol=1e-5), \
                f"Expected energy -10.53, got {energy}"

            # Check that no warning about unsuccessful completion was logged
            log_contents = log_stream.getvalue()
            assert "does not indicate successful completion" not in log_contents, \
                "No warning about unsuccessful completion should be logged for successful runs"

            print(f"Successful completion (no warning) test passed")
        finally:
            logger.removeHandler(handler)



if __name__ == "__main__":
    test_vasp_scf_energy()
    test_vasp_read_total_energy_static()
    test_vasp_energy_write_dat()
    test_vasp_energy_write_lmdb()
    test_vasp_multiple_energies_warning()
    test_vasp_unsuccessful_completion_warning()
    test_vasp_successful_completion_no_warning()
    print("\nAll VASP energy extraction tests passed!")
