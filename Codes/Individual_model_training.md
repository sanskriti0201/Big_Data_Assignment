# Individual Model Training


```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lower, udf, rand
from pyspark.sql.types import FloatType, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import pandas as pd
```


```python
spark = SparkSession.builder.appName("BRFSS_Model_Training").getOrCreate()

# Load filtered data
df = spark.read.option("header", True).csv(r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\original_filtered_30_39_csv")

```


```python
selected_cols = ["has_chronic_illness", "age", "bmi_category", "smoking_stat", 
                 "income_level", "physical_health", "employment_status",
                 "medical_cost_issue", "physical_activity", "housing", "drinking_stat", "marital_status_grouped",
                 "bp_diagnosed", "cholesterol_diagnosed", "chd_diagnosed", "depr_diagnosed", "diab_diagnosed"]
filtered_df = df.dropna(subset=selected_cols)
filtered_df.count()
```


```python
filtered_df.show(5)
```


```python
# Feature groups
physical_cols = ["bmi_category", "physical_activity", "physical_health", "drinking_stat", "smoking_stat"]
string_cols_phys = physical_cols

socio_eco_cols = ["income_level", "employment_status", "medical_cost_issue", "housing", "marital_status_grouped"]
string_cols_socio = socio_eco_cols 

both_cols = physical_cols + socio_eco_cols
string_cols_both = string_cols_phys + socio_eco_cols

chronic_labels = ["bp_diagnosed", "cholesterol_diagnosed", "chd_diagnosed", "depr_diagnosed", "diab_diagnosed"]
```


```python
def balance_dataset_string_labels(df, label_col):
    # Normalize to lowercase for consistent filtering
    df = df.withColumn(label_col, lower(col(label_col)))

    pos_df = df.filter(col(label_col) == "yes")
    neg_df = df.filter(col(label_col) == "no")

    count_pos = pos_df.count()
    count_neg = neg_df.count()

    if count_pos == 0 or count_neg == 0:
        print(f"Warning: No 'yes' or 'no' samples for {label_col}. Skipping balancing.")
        return df

    if count_pos < count_neg:
        neg_sampled = neg_df.sample(withReplacement=False, fraction=count_pos / count_neg, seed=42)
        balanced_df = pos_df.union(neg_sampled)
    else:
        pos_sampled = pos_df.sample(withReplacement=False, fraction=count_neg / count_pos, seed=42)
        balanced_df = pos_sampled.union(neg_df)

    return balanced_df

```


```python
get_prob = udf(lambda v: float(v[1]), DoubleType())

def train_evaluate_lr(train_df,test_df, feature_cols, string_cols, label_col, model_name):
    name_clean = model_name.replace(" ", "_").lower()
    prob_col = name_clean + "_prob"
    pred_col = name_clean + "_pred"

    indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx") for c in string_cols]
    assembler_inputs = [c + "_idx" if c in string_cols else c for c in feature_cols]
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
    
    lr = LogisticRegression(featuresCol="features", labelCol=label_col)
    pipeline = Pipeline(stages=indexers + [assembler, lr])
    model = pipeline.fit(train_df)
    preds = model.transform(test_df)
    
    evaluator_acc = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="f1")
    evaluator_auc = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    evaluator_precision = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="precisionByLabel")
    
    acc = evaluator_acc.evaluate(preds)
    f1 = evaluator_f1.evaluate(preds)
    auc = evaluator_auc.evaluate(preds)
    precision = evaluator_precision.evaluate(preds)
    
    preds = preds.withColumn(prob_col, get_prob("probability"))
    preds = preds.withColumnRenamed("prediction", pred_col)

    return acc, f1, auc, precision, preds.select("unique_id", prob_col, pred_col)
    
```


```python
individual_results = []
probs_df_individual = None

original_df = filtered_df  

for label in chronic_labels:
    print(f"\n Processing label: {label}")

    # Normalize to lowercase
    df_temp = original_df.withColumn(label, lower(col(label)))
    balanced_df = balance_dataset_string_labels(df_temp, label)

    if balanced_df.filter((col(label) == "yes") | (col(label) == "no")).count() == 0:
        print(f" Skipping {label} due to no positive/negative samples.")
        continue

    label_bin = label + "_bin"
    balanced_df = balanced_df.withColumn(label_bin, when(col(label) == "yes", 1).otherwise(0))

    train_df, test_df = balanced_df.randomSplit([0.8, 0.2], seed=42)
    
    # Generate unique model names per feature group and disease
    model_name_both = f"lr_both_{label}"
    model_name_phys = f"lr_physical_{label}"
    model_name_socio = f"lr_socio_eco_{label}"

    acc_both, f1_both, auc_both, precision_both, preds_both = train_evaluate_lr(train_df, test_df, both_cols, string_cols_both, label_bin, model_name_both)
    acc_phys, f1_phys, auc_phys, precision_phys, preds_phys = train_evaluate_lr(train_df, test_df, physical_cols, string_cols_phys, label_bin, model_name_phys)
    acc_socio, f1_socio, auc_socio, precision_socio, preds_socio = train_evaluate_lr(train_df, test_df, socio_eco_cols, string_cols_socio, label_bin, model_name_socio)

    print(f" {label} - Both: acc={acc_both:.4f}, f1={f1_both:.4f}, auc={auc_both:.4f}, precision={precision_both:.4f}")
    print(f"              Physical: acc={acc_phys:.4f}, f1={f1_phys:.4f}, auc={auc_phys:.4f}, precision={precision_phys:.4f}" )
    print(f"              Socio-economic: acc={acc_socio:.4f}, f1={f1_socio:.4f}, auc={auc_socio:.4f}, precision={precision_socio:.4f}")

    individual_results.append({
        "disease": label,
        "acc_both": acc_both,
        "f1_both": f1_both,
        "auc_both": auc_both,
        "precision_both": precision_both,
        "acc_phys": acc_phys,
        "f1_phys": f1_phys,
        "auc_phys": auc_phys,
        "precision_phys": precision_phys,
        "acc_socio": acc_socio,
        "f1_socio": f1_socio,
        "auc_socio": auc_socio,
        "precision_socio": precision_socio
    })

    # Join all three prediction sets
    for preds_df in [preds_both, preds_phys, preds_socio]:
        if probs_df_individual is None:
            probs_df_individual = preds_df
        else:
            probs_df_individual = probs_df_individual.join(preds_df, on="unique_id", how="left")

```


```python
probs_df_individual.show(5)
```


```python
export_path = r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\New_data\lr_probs_individual.csv"
probs_df_individual.toPandas().to_csv(export_path, index=False)
print(f"Exported predicted probabilities")
```


```python
# Convert list of dicts to DataFrame
individual_results = pd.DataFrame(individual_results)

# Define export path
results_export_path = r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\New_data\individual_model_evaluation_results.csv"

# Export to CSV without index
individual_results.to_csv(results_export_path, index=False)

print(f"Exported model evaluation results to {results_export_path}")

```


```python
spark.stop()
```


```python

```
