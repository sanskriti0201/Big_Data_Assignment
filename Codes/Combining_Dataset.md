# Load and combine Data and model results


```python
import pandas as pd
import os
```


```python
# Load all CSVs
original_path = r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\original_filtered_30_39_csv"
lr_individual_path = r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\New_data\lr_probs_individual.csv"
model_probs_path = r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\New_data\combined_models_probs.csv"
kmeans_path = r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\New_data\kmeans_cluster_results.csv"

```


```python
# Read CSV from folder
original_df = pd.concat(
    [pd.read_csv(os.path.join(original_path, f)) for f in os.listdir(original_path) if f.endswith(".csv")],
    ignore_index=True
)

# Load others
lr_df = pd.read_csv(lr_individual_path)
model_probs_df = pd.read_csv(model_probs_path)
kmeans_df = pd.read_csv(kmeans_path)

# Merge everything on `unique_id`
merged = original_df.merge(lr_df, on="unique_id", how="left") \
                    .merge(model_probs_df, on="unique_id", how="left") \
                    .merge(kmeans_df, on="unique_id", how="left")

# Save final CSV for Tableau
final_export_path = r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\New_data\new_merged_for_tableau.csv"
merged.to_csv(final_export_path, index=False)

print(f"Merged dataset saved to:\n{final_export_path}")

```

    Merged dataset saved to:
    C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\New_data\new_merged_for_tableau.csv
    


```python

```
