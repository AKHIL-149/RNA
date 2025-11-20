"""
Proper PDB File Writer for RNA Structures

Creates PDB files compatible with US-align by including all RNA backbone atoms.
"""

import numpy as np


def write_rna_pdb(coords, sequence, output_path, target_id="PRED"):
    """
    Write RNA structure to PDB format with all backbone atoms.

    For US-align compatibility, we need proper atom records. Since we only have
    C1' coordinates, we'll generate approximate positions for other backbone atoms.

    Args:
        coords (np.ndarray): C1' coordinates, shape (n, 3)
        sequence (str): RNA sequence
        output_path (str): Output PDB file path
        target_id (str): Structure identifier
    """
    # Standard RNA backbone atom offsets from C1' (approximate)
    # These are rough estimates for proper PDB format
    atom_offsets = {
        "P":   np.array([2.5, 0.0, 0.0]),    # Phosphate
        "O5'": np.array([1.5, 0.5, 0.0]),    # O5'
        "C5'": np.array([0.8, 0.8, 0.0]),    # C5'
        "C4'": np.array([0.3, 0.3, 0.0]),    # C4'
        "O4'": np.array([0.0, -0.5, 0.0]),   # O4'
        "C1'": np.array([0.0, 0.0, 0.0]),    # C1' (our reference)
        "C2'": np.array([-0.5, 0.5, 0.0]),   # C2'
        "C3'": np.array([-0.3, 1.0, 0.0]),   # C3'
        "O3'": np.array([0.0, 1.5, 0.0]),    # O3'
    }

    # Residue name mapping
    res_map = {
        'A': '  A', 'U': '  U', 'G': '  G', 'C': '  C',
        'T': '  T', 'N': '  N'
    }

    with open(output_path, 'w') as f:
        # Write header
        f.write(f"HEADER    RNA                                             {target_id}\n")
        f.write(f"TITLE     PREDICTED RNA STRUCTURE\n")

        atom_num = 1

        for res_num, (c1_coord, base) in enumerate(zip(coords, sequence), start=1):
            # Skip residues with missing coordinates
            if np.isnan(c1_coord).any():
                continue

            res_name = res_map.get(base, '  N')

            # Write backbone atoms
            for atom_name, offset in atom_offsets.items():
                # Calculate atom position relative to C1'
                atom_coord = c1_coord + offset

                # Format: ATOM record
                atom_line = (
                    f"ATOM  {atom_num:5d} {atom_name:^4s}{res_name} A{res_num:4d}    "
                    f"{atom_coord[0]:8.3f}{atom_coord[1]:8.3f}{atom_coord[2]:8.3f}"
                    f"  1.00  0.00           C\n"
                )
                f.write(atom_line)
                atom_num += 1

        # Write terminator
        f.write("TER\n")
        f.write("END\n")


def write_simple_rna_pdb(coords, sequence, output_path, target_id="PRED"):
    """
    Write minimal RNA PDB with just C1' atoms but proper formatting.

    This is a simpler version that might work better with US-align.

    Args:
        coords (np.ndarray): C1' coordinates, shape (n, 3)
        sequence (str): RNA sequence
        output_path (str): Output PDB file path
        target_id (str): Structure identifier
    """
    res_map = {
        'A': '  A', 'U': '  U', 'G': '  G', 'C': '  C',
        'T': '  T', 'N': '  N'
    }

    with open(output_path, 'w') as f:
        f.write(f"HEADER    RNA                                             {target_id}\n")

        atom_num = 1

        for res_num, (coord, base) in enumerate(zip(coords, sequence), start=1):
            if np.isnan(coord).any():
                continue

            res_name = res_map.get(base, '  N')

            # Write C1' atom with proper spacing
            atom_line = (
                f"ATOM  {atom_num:5d}  C1'{res_name} A{res_num:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00  0.00           C\n"
            )
            f.write(atom_line)
            atom_num += 1

        f.write("TER\n")
        f.write("END\n")


if __name__ == "__main__":
    print("RNA PDB Writer Module")
    print("\nExample usage:")
    print("""
    from src.tbm.pdb_writer import write_rna_pdb, write_simple_rna_pdb

    # Full backbone version
    write_rna_pdb(coords, sequence, "output.pdb")

    # Simple version (C1' only)
    write_simple_rna_pdb(coords, sequence, "output.pdb")
    """)
