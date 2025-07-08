# Sql functions for Data Analysis


```python
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as spark_sum, when, lower, lit, expr, row_number
from functools import reduce
from pyspark.sql.window import Window
```


```python
spark = SparkSession.builder.appName("BRFSS Analysis").getOrCreate()
original_df = spark.read.option("header", True).csv(r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\New_data\Data_Tableau\new_merged_for_tableau.csv")
original_df.createOrReplaceTempView("brfss_data")
```

## Converting dataframe to long format


```python
df_long = original_df
# List the disease columns
disease_cols = [
    "bp_diagnosed",
    "cholesterol_diagnosed",
    "chd_diagnosed",
    "depr_diagnosed",
    "diab_diagnosed"
]
query = """ 
SELECT
    unique_id,
    year,
    disease,
    LOWER(diagnosed) AS diagnosed,
    CASE WHEN LOWER(diagnosed) = 'yes' THEN 1 ELSE 0 END AS diagnosed_bin
FROM brfss_data
LATERAL VIEW STACK(5,
    'bp_diagnosed', bp_diagnosed,
    'cholesterol_diagnosed', cholesterol_diagnosed,
    'chd_diagnosed', chd_diagnosed,
    'depr_diagnosed', depr_diagnosed,
    'diab_diagnosed', diab_diagnosed
) AS disease, diagnosed
"""
df_long = spark.sql(query)
```


```python
df_long.show(10, truncate=False)
df_long.printSchema()
```


```python
# Define model info
models = [
    ("Logistic Regression", "logistic_regression_pred", "logistic_regression_prob"),
    ("Random Forest", "random_forest_pred", "random_forest_prob"),
    ("Gradient Boosted Trees", "gradient_boosted_trees_pred", "gradient_boosted_trees_prob")
]
# Create one DataFrame per model with proper columns
model_dfs = []
for name, pred_col, prob_col in models:
    model_df = original_df.select(
        "unique_id",
        "has_chronic_illness",
        col(pred_col).alias("predicted_label"),
        col(prob_col).alias("predicted_prob")
    ).withColumn("model", lit(name))
    model_dfs.append(model_df)
# Union all into one long-form dataframe
final_df = reduce(lambda a, b: a.unionByName(b), model_dfs)
# Reorde columns
final_df = final_df.select("unique_id", "model", "predicted_label", "predicted_prob", "has_chronic_illness")
```


```python
merged_df = df_long.join(final_df, on="unique_id", how="inner")
# Select only relevant columns
merged_df = merged_df.select(
    "unique_id", "year", "disease", "diagnosed_bin",
    "model", "predicted_label", "predicted_prob", "has_chronic_illness"
)
# Join disease diagnosis and model predictions on unique_id
merged_df = df_long.join(final_df, on="unique_id", how="inner")
# Select relevant columns
merged_df = merged_df.select(
    "unique_id", "year", "disease", "diagnosed_bin",
    "model", "predicted_label", "predicted_prob", "has_chronic_illness"
)
# Save full data as a single CSV file (Spark will create a folder)
output_path = r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\New_data\Data_Tableau\new_merged_actual_vs_predicted_full"
merged_df.coalesce(1).write.mode("overwrite").option("header", True).csv(output_path)

print("Full merged actual vs predicted file saved for Tableau.")
```

## Individual disease analysis
### Analysis using Physical and Socio-economic factor and both as well


```python
spark_df = original_df

# Disease labels
disease_cols = ["bp_diagnosed", "cholesterol_diagnosed", "chd_diagnosed", "depr_diagnosed", "diab_diagnosed"]

# Create binary labels
for c in disease_cols:
    spark_df = spark_df.withColumn(c, lower(col(c)))
    spark_df = spark_df.withColumn(c + "_bin", when(col(c) == "yes", 1).otherwise(0))

physical_factors = ["smoking_stat", "physical_activity", "physical_health", "drinking_stat", "bmi_category"]
socio_eco_factors = ["income_level", "employment_status", "housing", "medical_cost_issue", "marital_status_grouped"]
all_factors = physical_factors + socio_eco_factors

spark_df = spark_df.dropna(subset=all_factors)

from pyspark.sql.functions import rand

def balance_binary(df, label_col):
    positives = df.filter(col(label_col) == 1).orderBy(rand()).limit(1000)
    negatives = df.filter(col(label_col) == 0).orderBy(rand()).limit(1000)
    return positives.union(negatives)

# Main analysis
def analyze_factors(df, factors, diseases):
    results = []
    for disease in diseases:
        bin_col = disease + "_bin"
        balanced = balance_binary(df, bin_col)
        for factor in factors:
            res = (
                balanced.groupBy(factor)
                .agg(
                    count("*").alias("total"),
                    spark_sum(bin_col).alias("diagnosed_yes")
                )
                .withColumn("percentage_diagnosed", (col("diagnosed_yes") / col("total")) * 100)
                .withColumn("disease", lit(disease))
                .withColumn("factor", lit(factor))
                .withColumnRenamed(factor, "factor_level")
                .select("factor", "factor_level", "disease", "total", "diagnosed_yes", "percentage_diagnosed")
            )
            results.append(res)
    return reduce(lambda a, b: a.unionByName(b), results)


```


```python
result_all_df = analyze_factors(spark_df, all_factors, disease_cols)
```


```python
result_phy_df = analyze_factors(spark_df, physical_factors, disease_cols)
```


```python
result_socio_df = analyze_factors(spark_df, socio_eco_factors, disease_cols)
```


```python
result_all_df.show(5)
```


```python
result_all_df.toPandas().to_csv(r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\New_data\Data_Tableau\factor_analysis_all.csv", index=False)
result_phy_df.toPandas().to_csv(r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\New_data\Data_Tableau\factor_analysis_phy.csv", index=False)
result_socio_df.toPandas().to_csv(r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\New_data\Data_Tableau\factor_analysis_socio.csv", index=False)

print("All disease-factor interaction analysis saved!")

```

## Converting cluster to long format 


```python
original_pd = original_df.toPandas()

# Only relevant columns
id_cols = ['unique_id', 'cluster']
disease_cols = ['bp_diagnosed', 'cholesterol_diagnosed', 'chd_diagnosed', 'depr_diagnosed', 'diab_diagnosed']

# Melt into long format
long_df_cluster = original_pd.melt(id_vars=id_cols, value_vars=disease_cols,
                           var_name='disease', value_name='diagnosed')

# Optionally map 1/0 or Yes/No
long_df_cluster['diagnosed'] = long_df_cluster['diagnosed'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 1: 1, 0: 0})

long_df_cluster.to_csv(r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\New_data\Data_Tableau\long_cluster_disease.csv", index=False)

print("Cluster Melt Format saved")

```
