# Model Training 


```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, when, udf, lit
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import pandas as pd
```


```python
# Start Spark session
spark = SparkSession.builder.appName("BRFSS_Models_Test").getOrCreate()

# Load data
df = spark.read.option("header", True).csv(r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\original_filtered_30_39_csv")
```


```python
df.count()
```


```python
df.printSchema()
```


```python
# 1. Select and clean the data
selected_cols = ["has_chronic_illness", "age", "bmi", "bmi_category", "smoking_stat", 
                 "income_level", "physical_health", "employment_status",
                 "medical_cost_issue", "physical_activity", "housing", "drinking_stat", "marital_status_grouped",
                 "bp_diagnosed", "cholesterol_diagnosed", "chd_diagnosed", "depr_diagnosed", "diab_diagnosed"]
# Fix numeric types: cast age and bmi from string to float
df = df.withColumn("age", col("age").cast("float"))
df = df.withColumn("bmi", col("bmi").cast("float"))
df = df.withColumn("bmi", when(col("bmi") > 100, col("bmi") / 100).otherwise(col("bmi")))
df = df.withColumn("has_chronic_illness", col("has_chronic_illness").cast("int"))

filtered_df = df.dropna(subset=selected_cols)
```


```python
filtered_df.count()
```


```python
# 2. Balance the dataset
pos_df = filtered_df.filter(col("has_chronic_illness") == 1)
neg_df = filtered_df.filter(col("has_chronic_illness") == 0).orderBy(rand()).limit(pos_df.count())
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
evaluator_precision = MulticlassClassificationEvaluator(labelCol="has_chronic_illness", predictionCol="prediction", metricName="precisionByLabel")
```


```python
from itertools import chain
def get_expanded_feature_names(encoders, numeric_cols):
    cat_feature_names = []
    for encoder, base_name in zip(encoders, categorical_cols):
        try:
            size = encoder.categorySizes  # get number of one-hot encoded categories
        except:
            size = [1]  # fallback if info unavailable
        for i in range(len(size)):
            cat_feature_names.append(f"{base_name}_{i}")
    return cat_feature_names + numeric_cols

get_prob = udf(lambda v: float(v[1]), DoubleType())
train_df, test_df = balanced_df.randomSplit([0.8, 0.2], seed=42)

def evaluate_model(name, model, train_df, test_df, pipeline_stages):
    name_clean = name.replace(" ", "_").lower()
    prob_col = name_clean + "_prob"
    pred_col = name_clean + "_pred"

    # Train the pipeline
    pipeline = Pipeline(stages=pipeline_stages + [model])
    fitted = pipeline.fit(train_df)
    preds = fitted.transform(test_df)

    # Extract probability and rename prediction
    preds = preds.withColumn(prob_col, get_prob("probability"))
    preds = preds.withColumnRenamed("prediction", pred_col)

    # Evaluate using renamed prediction column
    metrics = {
        "model": name,
        "accuracy": evaluator_acc.evaluate(preds.withColumnRenamed(pred_col, "prediction")),
        "f1_score": evaluator_f1.evaluate(preds.withColumnRenamed(pred_col, "prediction")),
        "precision": evaluator_precision.evaluate(preds.withColumnRenamed(pred_col, "prediction")),
        "auc_roc": evaluator_auc.evaluate(preds),
        "auc_pr": evaluator_pr.evaluate(preds)
    }

    #Feature Importance
    expanded_feature_names = get_expanded_feature_names(encoders, numeric_cols)

    feature_importance = []
    try:
        if "Logistic Regression" in name:
            coeffs = fitted.stages[-1].coefficients.toArray()
            feature_importance = sorted(zip(expanded_feature_names, coeffs), key=lambda x: abs(x[1]), reverse=True)
        else:
            importances = fitted.stages[-1].featureImportances.toArray()
            feature_importance = sorted(zip(expanded_feature_names, importances), key=lambda x: x[1], reverse=True)

        # Print top 10 features
        print(f"\nTop Feature Importance for {name}:")
        for feat, val in feature_importance[:10]:
            print(f"{feat:30s}: {val:.5f}")
    except Exception as e:
        print(f"\n[Warning] Could not compute feature importance for {name}: {e}")

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
# Train-test split (80-20)

for name, clf in models:
    print(f" Training {name}... (without smote)")
    metrics, model_probs = evaluate_model(name, clf, train_df, test_df, pipeline_stages)
    results.append(metrics)
    print(f" {name} - Accuracy: {metrics['accuracy']:.5f}, F1: {metrics['f1_score']:.5f}, Precision: {metrics['precision']:.5f}, AUC-ROC: {metrics['auc_roc']:.5f}, AUC-PR: {metrics['auc_pr']:.5f}")
    
    if probs_df is None:
        probs_df = model_probs
    else:
        probs_df = probs_df.join(model_probs, on="unique_id", how="inner")


```

     Training Logistic Regression... (without smote)
    
    Top Feature Importance for Logistic Regression:
    medical_cost_issue_0          : -0.15936
    income_level_0                : 0.11912
    smoking_stat_0                : -0.11257
    physical_health_0             : 0.09359
    housing_0                     : 0.07336
    marital_status_grouped_0      : 0.06830
    physical_activity_0           : 0.05362
    bmi                           : -0.02018
    age                           : 0.02018
    drinking_stat_0               : 0.01365
     Logistic Regression - Accuracy: 0.63236, F1: 0.63113, Precision: 0.62296, AUC-ROC: 0.68704, AUC-PR: 0.69008
     Training Random Forest... (without smote)
    
    Top Feature Importance for Random Forest:
    medical_cost_issue_0          : 0.20059
    income_level_0                : 0.09053
    smoking_stat_0                : 0.05741
    housing_0                     : 0.03860
    physical_health_0             : 0.01481
    marital_status_grouped_0      : 0.01019
    physical_activity_0           : 0.00565
    drinking_stat_0               : 0.00028
    bmi_category_0                : 0.00009
    age                           : 0.00004
     Random Forest - Accuracy: 0.62529, F1: 0.62530, Precision: 0.62972, AUC-ROC: 0.67763, AUC-PR: 0.68087
     Training Gradient Boosted Trees... (without smote)
    
    Top Feature Importance for Gradient Boosted Trees:
    medical_cost_issue_0          : 0.08975
    smoking_stat_0                : 0.04349
    income_level_0                : 0.02155
    physical_health_0             : 0.01756
    housing_0                     : 0.01493
    bmi_category_0                : 0.01377
    age                           : 0.01372
    drinking_stat_0               : 0.01166
    physical_activity_0           : 0.01117
    marital_status_grouped_0      : 0.01045
     Gradient Boosted Trees - Accuracy: 0.63441, F1: 0.63346, Precision: 0.62633, AUC-ROC: 0.68609, AUC-PR: 0.68737
    


```python
output_path = r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\New_data\combined_models_probs.csv"
probs_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)
```


```python
final_results = pd.DataFrame(results)
final_results.to_csv(r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\combined_model_metrics.csv", index=False)
print("Model evaluation metrics saved.")

```

    Model evaluation metrics saved.
    


```python

```
