import pyBigWig as bw
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Open the BigWig file
path = "/home/vzscyborg/datasets/barret.bw"
fl = bw.open(path)

# Loop through all chromosomes
for chrom, chrom_length in fl.chroms().items():
    print(f"Processing {chrom}...")

    bin_size = 10000  
    positions = np.arange(0, chrom_length, bin_size)

    # Extract signal values from the BigWig file
    values = [
        fl.stats(chrom, pos, min(pos + bin_size, chrom_length), type="mean")[0] if fl.stats(chrom, pos, min(pos + bin_size, chrom_length), type="mean")[0] is not None else 0
        for pos in positions
    ]

    # Skip chromosome if it has no valid data or too few points
    if len(values) < 2:
        print(f"Skipping {chrom} (not enough data points).")
        continue

    # Convert to NumPy array
    X = np.array(values).reshape(-1, 1)

    # Standardize the values
    X_scaled = StandardScaler().fit_transform(X)

    # Generate labels (simulated: 50% enhancers, 50% non-enhancers)
    labels = np.random.choice([0, 1], size=len(X_scaled), replace=True)
    
    # Ensure at least two classes for LDA
    if len(set(labels)) == 1:
        labels[: len(labels) // 2] = 0
        labels[len(labels) // 2 :] = 1

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, labels, test_size=0.2, random_state=42
    )

    # Skip if train set is too small for LDA
    if len(X_train) < 2 or len(X_test) < 2:
        print(f"Skipping {chrom} (not enough training samples for LDA).")
        continue

    # Apply LDA
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    # Predict labels
    y_pred = lda.predict(X_test)

    # Evaluate LDA performance
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Accuracy for {chrom}: {accuracy:.4f}")
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # Scatter plot for visualization
    plt.figure(figsize=(8, 5))
    y_jitter = np.random.uniform(-0.1, 0.1, size=X_test_lda.shape)

    plt.scatter(X_test_lda, y_jitter, c=y_test, cmap="coolwarm", alpha=0.7)
    plt.xlabel("LDA Component 1")
    plt.ylabel("Random Jitter")
    plt.title(f"LDA of Genomic Signals in {chrom} (Accuracy: {accuracy:.4f})")
    plt.colorbar(label="Class Labels (0 = Non-enhancer, 1 = Enhancer)")

    # Save before plt.show()
    output_path = f"lda_plot_{chrom}.png"
    #plt.savefig(output_path, dpi=300, bbox_inches="tight")

    #plt.show()  # Show the plot after saving
    plt.pause(1)  # Pause for better viewing

    print(f"Saved LDA plot for {chrom} as {output_path}")

fl.close()  # Close the file when done
