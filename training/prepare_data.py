
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer


def main():
    dataset = load_breast_cancer()

    # Convert feature names into CSV-friendly column names
    feature_names = [
        name.replace(" ", "_")
        for name in dataset.feature_names
    ]

    df = pd.DataFrame(
        dataset.data,
        columns=feature_names
    )

    # Original target: 0 = malignant, 1 = benign
    # Required target: 1 = malignant, 0 = benign
    df["target"] = (dataset.target == 0).astype(int)

    # Validate dataset
    assert df.shape[1] == 31, "Dataset must contain 30 features and target"
    assert not df.isnull().any().any(), "Dataset contains missing values"
    assert df["target"].nunique() == 2, "Target must contain two classes"

    output_path = Path("data/raw/breast_cancer.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("Dataset created successfully!")
    print(f"Rows: {len(df)}")
    print(f"Features: {df.shape[1] - 1}")
    print(f"Columns: {df.shape[1]}")
    print(f"Malignant cases: {(df['target'] == 1).sum()}")
    print(f"Benign cases: {(df['target'] == 0).sum()}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

