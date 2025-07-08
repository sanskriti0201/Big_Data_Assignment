# K-Means Clustering


```python
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.sql.functions import col
```


```python
# Start Spark session
spark = SparkSession.builder.appName("BRFSS_kMeans").getOrCreate()

# Load data
df = spark.read.option("header", True).csv(r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\original_filtered_30_39_csv")
```


```python
df.printSchema()
```


```python
df = df.withColumn("age", col("age").cast("double"))

# Define features
categorical_cols = ["smoking_stat", "physical_activity","physical_health", "drinking_stat", "bmi_category",
                    "employment_status", "medical_cost_issue", "housing", "income_level", "marital_status_grouped"]
```


```python
# Index all categorical columns
for c in categorical_cols:
    indexer = StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
    df = indexer.fit(df).transform(df)

# Assemble indexed categorical features + numerical features
feature_cols = [c + "_idx" for c in categorical_cols] + ["age"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_unscaled")
df = assembler.transform(df)

# Apply StandardScaler
scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", withMean=True, withStd=True)
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)
```


```python
# KMeans clustering with k=2 (healthy vs unhealthy clusters)
kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=2, seed=42)
model = kmeans.fit(df)

# Assign clusters
predictions = model.transform(df)

# Show cluster centers
print("Cluster Centers:")
for i, center in enumerate(model.clusterCenters()):
    print(f"Cluster {i}: {center}")
```


```python
# Cluster distribution by chronic illness label
predictions.groupBy("cluster", "has_chronic_illness_label").count().show()
```


```python
predictions.groupBy("cluster").mean("age").show()
predictions.groupBy("cluster").count().show()
```


```python
# Export results 
export_cols = ["unique_id", "cluster"] 
export_df = predictions.select(*export_cols)

export_path = r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\New_data\kmeans_cluster_results.csv"
export_df.toPandas().to_csv(export_path, index=False)

print(f"Exported clustering results to {export_path}")

```


```python

```
