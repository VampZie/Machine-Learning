import pandas as pd
from Bio import PDB

pdb_file = "/home/vzscyborg/datasets/1rgs.pdb"
csv_file = "/home/vzscyborg/datasets/pdb_atoms.csv"

parser = PDB.PDBParser(QUIET=True)
structure = parser.get_structure("PDB_structure", pdb_file)

atom_data = []

for model in structure:
    for chain in model:
        for residue in chain:
            for atom in residue:
                atom_data.append([
                    atom.get_serial_number(),
                    atom.get_name(),
                    residue.get_resname(),
                    chain.get_id(),
                    residue.get_id()[1],
                    atom.get_coord()[0],
                    atom.get_coord()[1],
                    atom.get_coord()[2],
                    atom.get_occupancy(),
                    atom.get_bfactor(),
                    atom.element
                ])

df = pd.DataFrame(atom_data, columns=["Atom Serial", "Atom Name", "Residue Name", "Chain ID",
                                      "Residue Seq", "X", "Y", "Z", "Occupancy", "Temp Factor", "Element"])

df.to_csv(csv_file, index=False)

print(f"CSV file '{csv_file}' has been successfully created!")
