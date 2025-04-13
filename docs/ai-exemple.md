```python
# Install Dependencies
!pip install --upgrade pip
!pip install pandas scikit-learn --quiet

# Imports
import os
import time
import pandas as pd
import numpy as np
from typing import List
from pandas.errors import ParserError
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

# Dataset Loading Function
def load_datasets_with_skipping(dataset_dir):
    file_list = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.endswith('.csv') and os.path.isfile(os.path.join(dataset_dir, f))
    ]

    dfs = []
    for file_path in file_list:
        print(f"\nProcessing: {os.path.basename(file_path)}")
        skipped_rows = []

        try:
            df = pd.read_csv(
                file_path,
                on_bad_lines=lambda bad_line: skipped_rows.append(bad_line) or None,
                engine='python',
                dtype=str,
                encoding_errors='replace'
            )
            print(f"Loaded {len(df)} rows")
            if skipped_rows:
                print(f"Skipped {len(skipped_rows)} bad rows | Example: {skipped_rows[0][:100]}...")

            dfs.append(df)
        except Exception as e:
            print(f"FAILED: {str(e)}")

    return dfs

# Column Diagnostic Check
dataset_dir = 'datasets'

# Check first 3 files' columns
print("\n=== FIRST 3 FILES COLUMN CHECK ===")
sample_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')][:3]
for file in sample_files:
    try:
        df_sample = pd.read_csv(os.path.join(dataset_dir, file), nrows=1)
        print(f"{file}: Columns → {df_sample.columns.tolist()}")
    except Exception as e:
        print(f"{file}: Error reading → {str(e)}")

# Dynamic Preprocessing - Optimized

# List of common target column names (prioritized order)
TARGET_COL = ["entity_value","Threat","label","AffectedProducts","submitted","time_ts","date"]

# Load datasets
dfs = load_datasets_with_skipping(dataset_dir)
if not dfs:
    raise ValueError("No datasets loaded - check directory path!")

processed_dfs = []
skipped_count = 0

for idx, df in enumerate(dfs):
    print(f"\nProcessing Dataset {idx+1}/{len(dfs)}")

    # Skip empty datasets
    if df.empty:
        print("⚠️ Empty dataframe - skipped")
        skipped_count += 1
        continue

    # --- Dynamic Target Detection (improved) ---
    target_col = None

    # 1. Check for exact matches in common target names
    for candidate in TARGET_COL:
        if candidate in df.columns:
            target_col = candidate
            print(f"Found target column: '{target_col}'")
            break

    # 2. Check for text columns with classification potential
    if target_col is None:
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            n_unique = df[col].nunique()
            if 2 <= n_unique <= 20:  # Good classification target range
                target_col = col
                print(f"Selected text column as target: '{target_col}' with {n_unique} unique values")
                break

    # 3. Fallback: Find column with reasonable number of unique values
    if target_col is None:
        unique_counts = df.nunique()
        # Better criteria: 2-20 unique values but exclude columns with too high % of NaN
        potential_targets = [
            col for col in unique_counts.index
            if 2 <= unique_counts[col] <= 20 and df[col].isna().mean() < 0.3
        ]

        if potential_targets:
            target_col = potential_targets[0]
            print(f"⚠️ Assuming '{target_col}' as target ({unique_counts[target_col]} unique values)")
        else:
            # 4. Last resort: Use last column
            target_col = df.columns[-1]
            print(f"⚠️ No clear target found. Using last column '{target_col}'")

    # --- Preprocessing (improved) ---
    try:
        # Rename target column for consistency
        df = df.rename(columns={target_col: 'target'})

        # Handle special case: If target is binary numeric (0/1), keep it as is
        is_binary_numeric = (df['target'].nunique() == 2 and
                             pd.api.types.is_numeric_dtype(df['target']) and
                             set(df['target'].unique()).issubset({0, 1, 0.0, 1.0}))

        # Extract features (everything except target)
        X = df.drop('target', axis=1)
        y = df['target']

        # Drop columns with too many missing values
        X = X.loc[:, X.isna().mean() < 0.5]  # Keep columns with <50% missing values

        if X.empty:
            print("⚠️ All feature columns had too many missing values - skipped")
            skipped_count += 1
            continue

        # Intelligent feature type conversion
        for col in X.columns:
            # Try numeric conversion first
            if not pd.api.types.is_numeric_dtype(X[col]):
                try:
                    X[col] = pd.to_numeric(X[col], errors='raise')
                except:
                    # If failed, try categorical encoding for object columns with low cardinality
                    if pd.api.types.is_object_dtype(X[col]) and X[col].nunique() <= 20:
                        X[col] = X[col].astype('category').cat.codes
                    else:
                        # Drop high-cardinality text columns that can't be converted
                        X = X.drop(col, axis=1)
                        print(f"Dropped non-convertible column '{col}'")

        # Handle missing values intelligently
        for col in X.columns:
            if X[col].isna().any():
                if pd.api.types.is_numeric_dtype(X[col]):
                    # Use median for numeric columns (more robust than mean)
                    X[col] = X[col].fillna(X[col].median())
                else:
                    # Use most frequent value for categorical
                    X[col] = X[col].fillna(X[col].mode()[0])

        # Get rows without any remaining NaN values
        valid_mask = X.notna().all(axis=1)
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()

        if len(X_clean) < 10:  # Too few samples
            print(f"⚠️ Only {len(X_clean)} valid rows after cleaning - skipped")
            skipped_count += 1
            continue

        # Encode target labels if needed
        if not is_binary_numeric and not pd.api.types.is_numeric_dtype(y_clean):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_clean)
            print(f"Encoded {le.classes_.size} unique target values")
        else:
            y_encoded = y_clean

        # Construct final dataframe
        final_df = pd.concat([X_clean, pd.Series(y_encoded, name='target')], axis=1)

        # Report dataset stats
        print(f"✓ Processed dataset: {final_df.shape[0]} samples, {final_df.shape[1]-1} features")
        processed_dfs.append(final_df)

    except Exception as e:
        print(f"❌ Failed to process dataset {idx+1}: {str(e)}")
        skipped_count += 1
        continue

# Final check
if not processed_dfs:
    raise ValueError("❌ All datasets failed preprocessing")

print(f"\n✅ Success: {len(processed_dfs)} datasets ready for training ({skipped_count} skipped)")

# Show dataset information
print("\nProcessed Datasets:")
for i, df in enumerate(processed_dfs):
    print(f"Dataset {i+1}: {df.shape[0]} samples, {df.shape[1]-1} features")
    if i == 0:  # For the first dataset, show features
        print("Features:", df.drop('target', axis=1).columns.tolist())
        print("Target distribution:", df['target'].value_counts().to_dict())

# Robust Feature Consistency Check

# Skip check if only one dataset
if len(processed_dfs) <= 1:
    print("✅ Only one dataset present - skipping feature consistency check")
else:
    # Get reference features from first dataset
    ref_features = set(processed_dfs[0].drop('target', axis=1).columns)
    feature_counts = {}

    # Count feature occurrences across all datasets
    for feature in ref_features:
        feature_counts[feature] = 1

    # Check other datasets and count feature occurrences
    for idx, df in enumerate(processed_dfs[1:], start=2):
        current_features = set(df.drop('target', axis=1).columns)
        for feature in current_features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

    # Find features that occur in at least 50% of datasets
    total_datasets = len(processed_dfs)
    frequent_features = [f for f, count in feature_counts.items()
                         if count >= total_datasets * 0.5]

    if not frequent_features:
        print("⚠️ No common features across datasets. Using all features from the first dataset.")
        # Use first dataset's features as reference
        master_features = list(ref_features)
    else:
        print(f"🔍 Found {len(frequent_features)} features that appear in at least 50% of datasets.")
        master_features = frequent_features

    # Standardize all datasets to use these features
    for i, df in enumerate(processed_dfs):
        # Get current features
        current_features = set(df.drop('target', axis=1).columns)

        # Add missing features with default values (zeros)
        for feature in set(master_features) - current_features:
            processed_dfs[i][feature] = 0

        # Select only the needed columns (in same order)
        processed_dfs[i] = processed_dfs[i][master_features + ['target']]

    print(f"✅ All datasets standardized to use {len(master_features)} features")

# Colab‑ready incremental training with SGDClassifier

def incremental_train(
    processed_dfs: List[pd.DataFrame],
    target_col: str = 'target',
    verbose: bool = True
):
    """
    Incrementally train an SGDClassifier over multiple DataFrames.
    Each df must contain a `target_col` and at least one feature.
    """
    if not processed_dfs:
        raise ValueError("No processed datasets available for training")

    # Filter out any dfs that only have the target column
    valid_dfs = [df for df in processed_dfs if df.shape[1] > 1]
    if not valid_dfs:
        raise ValueError("No datasets with features available for training")
    if verbose:
        print(f"✔️  Found {len(valid_dfs)} valid dataset(s) with features.")

    # Gather all classes across datasets (two‑pass)
    all_classes = sorted({
        cls for df in valid_dfs
        for cls in df[target_col].unique()
    })
    classes_array = np.array(all_classes)
    if verbose:
        print(f"🔢  Total distinct classes: {len(classes_array)} -> {classes_array}")

    # Initialize scaler and model
    scaler = StandardScaler()
    model = SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=1e-4,
        max_iter=1,       # we call partial_fit ourselves
        tol=None,         # disable built‑in stopping
        learning_rate='optimal',
        class_weight='balanced',
        random_state=42
    )

    n_samples = 0
    start_time = time.time()

    # Loop with progress bar
    for idx, df in enumerate(tqdm(valid_dfs, desc="Incremental training")):
        try:
            X = df.drop(columns=target_col)
            y = df[target_col].values

            if X.shape[1] == 0:
                if verbose:
                    print(f"⚠️  Skipping chunk {idx+1}: no features")
                continue

            # Incrementally update scaler and transform
            scaler.partial_fit(X)
            X_scaled = scaler.transform(X)

            # First call to partial_fit must include classes
            if n_samples == 0:
                model.partial_fit(X_scaled, y, classes=classes_array)
            else:
                model.partial_fit(X_scaled, y)

            # Metrics & counters
            n_samples += len(y)
            acc = accuracy_score(y, model.predict(X_scaled))
            if verbose:
                print(f"  • Chunk {idx+1}: {len(y)} samples, {X.shape[1]} features → acc {acc:.4f}")

        except Exception as e:
            print(f"❌  Error on chunk {idx+1}: {e}")
            continue

    if n_samples == 0:
        raise RuntimeError("No samples were processed during training")

    total_time = time.time() - start_time
    print(f"\n✅  Training finished in {total_time:.2f}s on {n_samples} samples")
    print(f"   • Final scaler mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
    print(f"   • Final scaler scale range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")
    print(f"   • Classes seen: {classes_array}")

    return model, scaler
```