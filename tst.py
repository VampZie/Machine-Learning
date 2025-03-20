import pyBigWig as bw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load BigWig file
path = "/home/vzscyborg/datasets/barret.bw"
fl = bw.open(path)

# Choose chromosome
chrom = "chr1"
chrom_length = fl.chroms()[chrom]

# Define binning parameters
bin_size = 5000
num_bins = chrom_length // bin_size

# Get signal values
positions = np.arange(0, chrom_length, bin_size)
values = [fl.stats(chrom, pos, pos + bin_size, type="mean")[0] or 0 for pos in positions]

fl.close()

# Convert to DataFrame
df = pd.DataFrame({"Position": positions, "Signal": values})

# Plot heatmap using imshow()
plt.figure(figsize=(10, 5))
plt.imshow(df["Signal"].values.reshape(1, -1), aspect="auto", cmap="coolwarm", extent=[0, chrom_length, 0, 1])
plt.colorbar(label="Signal Intensity")
plt.title(f"Genomic Heatmap for {chrom}")
plt.xlabel("Genomic Position")
plt.yticks([])  # Hide y-axis since it's a single row
plt.show()
