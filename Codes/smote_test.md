# Model Training Using Smote


```python
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, when, udf, lit
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import functions as F
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
```


```python
# Start Spark session
spark = SparkSession.builder.appName("BRFSS_Models_Test").getOrCreate()

# Load data
df = spark.read.option("header", True).csv(r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\original_filtered_30_39_csv")

# Convert Spark DataFrame to Pandas 
pandas_df = df.toPandas()
```


```python
essential_features = ["has_chronic_illness", "age", "bmi", "bmi_category", "smoking_stat", 
                      "income_level", "physical_health", "employment_status",
                      "medical_cost_issue", "physical_activity", "housing", "drinking_stat", "marital_status_grouped"]
pandas_df = pandas_df.dropna(subset=essential_features)
```


```python
label_col = 'has_chronic_illness'
id_col = 'unique_id'

# Feature columns
feature_cols = [col for col in pandas_df.columns if col not in [label_col, id_col]]

# Encode categorical features
for col in feature_cols:
    if pandas_df[col].dtype == 'object':
        le = LabelEncoder()
        pandas_df[col] = le.fit_transform(pandas_df[col].astype(str))

# Encode label 
if pandas_df[label_col].dtype == 'object':
    le_label = LabelEncoder()
    pandas_df[label_col] = le_label.fit_transform(pandas_df[label_col].astype(str))

# Saving original unique IDs for later
original_ids = pandas_df[id_col].values
```


```python
# Preparing features and labels for SMOTE
X = pandas_df[feature_cols]
y = pandas_df[label_col]

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```


```python
# Converting unique_id to int
original_ids = pandas_df['unique_id'].astype(int).values

n_original = len(original_ids)
n_resampled = len(X_resampled)

new_ids = np.arange(n_resampled)
new_ids[:n_original] = original_ids  # keep original IDs
new_ids[n_original:] = np.arange(original_ids.max() + 1, original_ids.max() + 1 + (n_resampled - n_original))

```


```python
# Combine resampled features, labels, and IDs into one DataFrame
resampled_df = pd.DataFrame(X_resampled, columns=feature_cols)
resampled_df[label_col] = y_resampled
resampled_df[id_col] = new_ids

# Convert to Spark DataFrame
balanced_df = spark.createDataFrame(resampled_df)
```


```python
balanced_df.show()
```


```python
from pyspark.sql.functions import col, when

# Fix numeric types: cast age and bmi from string to float
balanced_df = balanced_df.withColumn("age", col("age").cast("float"))
balanced_df = balanced_df.withColumn("bmi", col("bmi").cast("float"))
balanced_df = balanced_df.withColumn("bmi", when(col("bmi") > 100, col("bmi") / 100).otherwise(col("bmi")))
balanced_df = balanced_df.withColumn("has_chronic_illness", col("has_chronic_illness").cast("int"))
```


```python
# 2. Balance the dataset
pos_df = balanced_df.filter(col("has_chronic_illness") == 1)
neg_df = balanced_df.filter(col("has_chronic_illness") == 0).orderBy(rand()).limit(pos_df.count())
balanced_df = pos_df.union(neg_df)

# 3. Define categorical and numeric columns
categorical_cols = ["smoking_stat", "physical_activity", "physical_health", "drinking_stat", "bmi_category",
                    "employment_status", "medical_cost_issue", "housing", "income_level", "marital_status_grouped"]

numeric_cols = ["age", "bmi"]
```


```python
# 4. Index and encode categorical columns
indexers = [StringIndexer(inputCol=col, outputCol=col + "_idx", handleInvalid="keep") for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=col + "_idx", outputCol=col + "_vec") for col in categorical_cols]

# 5. Vector assembler and scaler
feature_cols = [col + "_vec" for col in categorical_cols] + numeric_cols
assembler = VectorAssembler(inputCols=feature_cols, outputCol="unscaled_features")
scaler = StandardScaler(inputCol="unscaled_features", outputCol="features")
```


```python
# 6. Define evaluators
evaluator_auc = BinaryClassificationEvaluator(labelCol="has_chronic_illness", metricName="areaUnderROC")
evaluator_pr = BinaryClassificationEvaluator(labelCol="has_chronic_illness", metricName="areaUnderPR")
evaluator_acc = MulticlassClassificationEvaluator(labelCol="has_chronic_illness", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="has_chronic_illness", predictionCol="prediction", metricName="f1")
```


```python
get_prob = udf(lambda v: float(v[1]), DoubleType())
def evaluate_model(name, model, df, pipeline_stages):
    name_clean = name.replace(" ", "_").lower()
    prob_col = name_clean + "_prob"
    pred_col = name_clean + "_pred"

    pipeline = Pipeline(stages=pipeline_stages + [model])
    fitted = pipeline.fit(df)
    preds = fitted.transform(df)

    # Extract probability and rename prediction
    preds = preds.withColumn(prob_col, get_prob("probability"))
    preds = preds.withColumnRenamed("prediction", pred_col)

    # Evaluate using default prediction
    metrics = {
    "model": name,
    "accuracy": evaluator_acc.evaluate(preds.withColumnRenamed(pred_col, "prediction")),
    "f1_score": evaluator_f1.evaluate(preds.withColumnRenamed(pred_col, "prediction")),
    "auc_roc": evaluator_auc.evaluate(preds),
    "auc_pr": evaluator_pr.evaluate(preds)
}
    return metrics, preds.select("unique_id", prob_col, pred_col)
```


```python
# 9. Prepare pipeline stages
pipeline_stages = indexers + encoders + [assembler, scaler]

# 10. Train and evaluate models
results = []
probs_df = None

models = [
    ("Logistic Regression", LogisticRegression(labelCol="has_chronic_illness", featuresCol="features")),
    ("Random Forest", RandomForestClassifier(labelCol="has_chronic_illness", featuresCol="features")),
    ("Gradient Boosted Trees", GBTClassifier(labelCol="has_chronic_illness", featuresCol="features", maxIter=50))
]
```


```python
for name, clf in models:
    print(f" Training {name}...")
    metrics, model_probs = evaluate_model(name, clf, balanced_df, pipeline_stages)
    results.append(metrics)
    print(f" {name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}, AUC-PR: {metrics['auc_pr']:.4f}")

    if probs_df is None:
        probs_df = model_probs
    else:
        probs_df = probs_df.join(model_probs, on="unique_id", how="inner")

```


```python
probs_df = probs_df.cache()
```


```python
output_path = r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\New_data\probs_raw"
probs_df.write.mode("overwrite").option("header", True).csv(output_path)
print("Saved directly from Spark to:", output_path)

```


```python
import os
# Count total records
total_rows = probs_df.count()
batch_size = 10000 
num_batches = (total_rows // batch_size) + 1

from pyspark.sql.functions import monotonically_increasing_id

# Add an index column for partitioning manually
probs_df = probs_df.withColumn("row_id", monotonically_increasing_id())

output_dir = r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\New_data\combined_models_probs"

for i in range(num_batches):
    start = i * batch_size
    end = (i + 1) * batch_size
    batch_df = probs_df.filter((probs_df.row_id >= start) & (probs_df.row_id < end)).drop("row_id")

    # Save each chunk
    batch_path = os.path.join(output_dir, f"part_{i+1}")
    batch_df.write.mode("overwrite").option("header", True).csv(batch_path)

    print(f"Saved batch {i+1} to {batch_path}")

```


```python
# Convert list of dicts to DataFrame
results = pd.DataFrame(results)

# Define export path
results_export_path = r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\New_data\combined_model_evaluation_results.csv"

# Export to CSV without index
results.to_csv(results_export_path, index=False)

print(f"Exported model evaluation results to {results_export_path}")

```
