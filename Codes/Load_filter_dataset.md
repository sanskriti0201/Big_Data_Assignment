# Load, filter and Map Data


```python
import pandas as pd
import pyreadstat
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
```


```python
#Spark session
spark = SparkSession.builder.appName("BRFSS_Data_Multiyear").getOrCreate()
```


```python
# file paths and variable mappings by year
files_info = {
    2017: {
        "path": r"C:\Users\sansk\Downloads\LLCP2017XPT\LLCP2017.XPT",
        "cols": {
            '_AGE80': 'age', 'SEX': 'sex', 
            'BPHIGH4': 'bp', 'TOLDHI2': 'chol', 'CVDCRHD4': 'chd', 'DIABETE3': 'diab', 'ADDEPEV2': 'depr', 
            '_SMOKER3': 'smk_stat', '_DRNKWEK': 'drnk_stat', '_BMI5': 'bmi', 'EXERANY2': 'phy_act', 'PHYSHLTH': 'phy_hlth',
            '_INCOMG': 'income', 'EMPLOY1': 'employment', 'MEDCOST': 'medcost', 'RENTHOM1': 'housing_stat', 'MARITAL': 'marital_status'
        }
    },
    2019: {
        "path": r"C:\Users\sansk\Downloads\LLCP2019XPT\LLCP2019.XPT",
        "cols": {
            '_AGE80': 'age', '_SEX': 'sex', 
            'BPHIGH4': 'bp', 'TOLDHI2': 'chol', 'CVDCRHD4': 'chd', 'DIABETE4': 'diab', 'ADDEPEV3': 'depr', 
            '_SMOKER3': 'smk_stat', '_DRNKWK1': 'drnk_stat', '_BMI5': 'bmi', 'EXERANY2': 'phy_act', 'PHYSHLTH': 'phy_hlth',
            '_INCOMG': 'income', 'EMPLOY1': 'employment', 'MEDCOST': 'medcost', 'RENTHOM1': 'housing_stat', 'MARITAL': 'marital_status'
        }
    },
    2021: {
        "path": r"C:\Users\sansk\Downloads\LLCP2021XPT\LLCP2021.XPT",
       "cols": {
            '_AGE80': 'age', '_SEX': 'sex', 
            'BPHIGH6': 'bp', 'TOLDHI3': 'chol', 'CVDCRHD4': 'chd', 'DIABETE4': 'diab', 'ADDEPEV3': 'depr', 
            '_SMOKER3': 'smk_stat', '_DRNKWK1': 'drnk_stat', '_BMI5': 'bmi', 'EXERANY2': 'phy_act', 'PHYSHLTH': 'phy_hlth',
            '_INCOMG1': 'income', 'EMPLOY1': 'employment', 'MEDCOST1': 'medcost', 'RENTHOM1': 'housing_stat', 'MARITAL': 'marital_status'
        }
    },
    2023: {
        "path": r"C:\Users\sansk\Downloads\LLCP2023XPT\LLCP2023.XPT",
        "cols": {
            '_AGE80': 'age', '_SEX': 'sex', 
            'BPHIGH6': 'bp', 'TOLDHI3': 'chol', 'CVDCRHD4': 'chd', 'DIABETE4': 'diab', 'ADDEPEV3': 'depr', 
            '_SMOKER3': 'smk_stat', '_DRNKWK2': 'drnk_stat', '_BMI5': 'bmi', 'EXERANY2': 'phy_act', 'PHYSHLTH': 'phy_hlth',
            '_INCOMG1': 'income', 'EMPLOY1': 'employment', 'MEDCOST1': 'medcost', 'RENTHOM1': 'housing_stat', 'MARITAL': 'marital_status'
        }
    }
}
```


```python
def load_and_process_year(year, info, all_final_cols):
    print(f"Loading year: {year}")
    try:
        # Conditional read: pandas.read_sas for 2023 only
        if year == 2023:
            df_pd = pd.read_sas(info["path"], format="xport")
        else:
            df_pd, _ = pyreadstat.read_xport(info["path"])

        rename_map = info["cols"]
        available_cols = [col for col in rename_map if col in df_pd.columns]
        df_pd = df_pd[available_cols].rename(columns={k: rename_map[k] for k in available_cols})
        
        # Add missing columns as None
        for col in all_final_cols:
            if col not in df_pd.columns:
                df_pd[col] = None

        # Reorder columns
        df_pd = df_pd[all_final_cols]

        # Convert all to string and fill NaNs
        for c in df_pd.columns:
            df_pd[c] = df_pd[c].astype('str').fillna('')

        # Define schema
        schema = StructType([StructField(col_name, StringType(), True) for col_name in all_final_cols])

        df_spark = spark.createDataFrame(df_pd, schema=schema).withColumn("year", lit(year))
        print(f"Successfully loaded year {year}, records: {df_spark.count()}")

        return df_spark

    except Exception as e:
        print(f"Failed to load year {year}: {e}")
        return None

```


```python
#Calculate all columns needed for consistent schema
all_final_cols = set()
for info in files_info.values():
    all_final_cols.update(info['cols'].values())
all_final_cols = sorted(list(all_final_cols))

```


```python
#Load and combine all years
df_list = []
for year, info in files_info.items():
    df_year = load_and_process_year(year, info, all_final_cols)
    if df_year is not None:
        df_list.append(df_year)
```


```python
# Union all DataFrames into one combined DataFrame
combined_df = df_list[0]
for df in df_list[1:]:
    combined_df = combined_df.unionByName(df)

print(f"\nCombined DataFrame row count: {combined_df.count()}")
```


```python
combined_df.show(5)
```

Crearing DataSet and Applying Filters.


```python
#Step1: Create original dataset: Filter combined_df by age 30-39 
original_df = combined_df.filter((col('age') >= 30) & (col('age') <= 39))
print(f"Original DataFrame (age 30-39) count (including NaNs): {original_df.count()}")

window = Window.orderBy(lit(1))  # No specific column needed
original_df = original_df.withColumn("unique_id", row_number().over(window) - 1)
original_df.show(5)
```

    Original DataFrame (age 30-39) count (including NaNs): 205937
    


```python
original_df.printSchema()
```


```python
from pyspark.sql.functions import when, col, lower

# Cleaning: handle 'nan', 'null', 'none' and cast float strings to int
def clean_and_cast_int(df, col_name):
    return df.withColumn(
        col_name,
        when(lower(col(col_name)).isin("nan", "null", "none"), None)
        .otherwise(col(col_name).cast("float").cast("int"))
    )

# Same logic for float columns
def clean_and_cast_float(df, col_name):
    return df.withColumn(
        col_name,
        when(lower(col(col_name)).isin("nan", "null", "none"), None)
        .otherwise(col(col_name).cast("float"))
    )

```


```python
int_columns = [
    "sex", "bp", "chol", "chd", "depr", "diab", "smk_stat", 
    "drnk_stat", "phy_act", "phy_hlth", "income", "employment",
    "housing_stat", "marital_status", "medcost"
]

for c in int_columns:
    original_df = clean_and_cast_int(original_df, c)

original_df = clean_and_cast_float(original_df, "bmi")

# Normalize BMI
original_df = original_df.withColumn(
    "bmi", when(col("bmi") > 100, col("bmi") / 100).otherwise(col("bmi"))
)

```


```python
from pyspark.sql.functions import when, col

original_df = original_df.withColumn(
    "sex_category",
    when(col("sex") == 1, "Male")
    .when(col("sex") == 2, "Female")
    .otherwise(None)
)

original_df = original_df.withColumn(
    "bp_diagnosed",
    when(col("bp") == 1, "Yes")
    .when(col("bp") == 3, "No")
    .otherwise(None)
)

original_df = original_df.withColumn(
    "cholesterol_diagnosed",
    when(col("chol") == 1, "Yes")
    .when(col("chol") == 2, "No")
    .otherwise(None)
)

original_df = original_df.withColumn(
    "chd_diagnosed",
    when(col("chd") == 1, "Yes")
    .when(col("chd") == 2, "No")
    .otherwise(None)
)

original_df = original_df.withColumn(
    "depr_diagnosed",
    when(col("depr") == 1, "Yes")
    .when(col("depr") == 2, "No")
    .otherwise(None)
)

original_df = original_df.withColumn(
    "diab_diagnosed",
    when(col("diab") == 1, "Yes")
    .when(col("diab") == 3, "No")
    .otherwise(None)
)

original_df = original_df.withColumn(
    "smoking_stat",
    when(col("smk_stat") == 1, "Active Smoker")
    .when(col("smk_stat") == 2, "Inactive Smoker")
    .when(col("smk_stat") == 3, "Former Smoker")
    .when(col("smk_stat") == 4, "Non Smoker")
    .otherwise(None)
)

original_df = original_df.withColumn(
    "drinking_stat",
    when(col("drnk_stat") == 0, "Non Drinker")
    .when((col("drnk_stat") >= 1) & (col("drnk_stat") <= 5), "Light Drinker")
    .when((col("drnk_stat") >= 6) & (col("drnk_stat") <= 14), "Moderate Drinker")
    .when((col("drnk_stat") > 14) & (col("drnk_stat") <= 98999), "Heavy Drinker")
    .otherwise(None)
)

# bmi fix - casted to float already
original_df = original_df.withColumn(
    "bmi",
    when(col("bmi") > 100, col("bmi") / 100).otherwise(col("bmi"))
)

original_df = original_df.withColumn(
    "bmi_category",
    when((col("bmi") == 99.99) | col("bmi").isNull(), None)
    .when(col("bmi") < 18.5, "Underweight")
    .when((col("bmi") >= 18.5) & (col("bmi") <= 24.9), "Optimum range")
    .when((col("bmi") >= 25) & (col("bmi") <= 29.9), "Overweight")
    .when((col("bmi") >= 30) & (col("bmi") <= 34.9), "Class I obesity")
    .when(col("bmi") >= 35, "Class II obesity")
    .otherwise(None)
)

original_df = original_df.withColumn(
    "physical_activity",
    when(col("phy_act") == 1, "Active")
    .when(col("phy_act") == 2, "Inactive")
    .otherwise(None)
)

original_df = original_df.withColumn(
    "physical_health",
    when((col("phy_hlth") == 88) | (col("phy_hlth") < 3), "Excellent")
    .when((col("phy_hlth") >= 3) & (col("phy_hlth") <= 7), "Good")
    .when((col("phy_hlth") > 7) & (col("phy_hlth") <= 13), "At risk")
    .when((col("phy_hlth") > 13) & (col("phy_hlth") < 77), "Poor")
    .when((col("phy_hlth") == 77) | (col("phy_hlth") == 99), None)
    .otherwise(None)
)

original_df = original_df.withColumn(
    "income_level",
    when(col("income") == 1, "Less than $10,000")
    .when(col("income") == 2, "$10,000 - $14,999")
    .when(col("income") == 3, "$15,000 - $19,999")
    .when(col("income") == 4, "$20,000 - $24,999")
    .when(col("income") == 5, "$25,000 - $34,999")
    .when(col("income") == 6, "$35,000 - $49,999")
    .when(col("income") == 7, "$50,000 - $74,999")
    .when(col("income") == 8, "$75,000 or more")
    .otherwise(None)
)

original_df = original_df.withColumn(
    "employment_status",
    when(col("employment") == 1, "Employed for wages")
    .when(col("employment") == 2, "Self-employed")
    .when(col("employment") == 3, "Out of work for 1 year or more")
    .when(col("employment") == 4, "Out of work for less than 1 year")
    .when(col("employment") == 5, "Homemaker")
    .when(col("employment") == 6, "Student")
    .when(col("employment") == 7, "Retired")
    .when(col("employment") == 8, "Unable to work")
    .otherwise(None)
)

original_df = original_df.withColumn(
    "medical_cost_issue",
    when(col("medcost") == 1, "Yes")
    .when(col("medcost") == 2, "No")
    .otherwise(None)
)

original_df = original_df.withColumn(
    "housing",
    when(col("housing_stat") == 1, "Own")
    .when(col("housing_stat") == 2, "Rent")
    .otherwise(None)
)

original_df = original_df.withColumn(
    "marital_status_grouped",
    when(col("marital_status") == 1, "Married")
    .when(col("marital_status") == 2, "Divorced")
    .when(col("marital_status") == 3, "Widowed")
    .when(col("marital_status") == 4, "Separated")
    .when(col("marital_status") == 5, "Never married")
    .when(col("marital_status") == 6, "Unmarried couple")
    .otherwise(None)
)
# Create 'has_chronic_illness' column with 1 if any of the 5 conditions are met
original_df = original_df.withColumn(
    "has_chronic_illness",
    when(
        (col("bp") == 1) |
        (col("chol") == 1) |
        (col("chd") == 1) |
        (col("depr") == 1) |
        (col("diab") == 1),
        1
    ).otherwise(0)
)

original_df = original_df.withColumn(
    "has_chronic_illness_label",
    when(col("has_chronic_illness") == 1, "Yes").otherwise("No")
)
```


```python
original_df.show(1)
```


```python
original_df.groupBy("year").count().orderBy("year").show()
```


```python
original_df.write.mode("overwrite").parquet(r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\original_filtered_30_39.parquet")
```


```python
original_df.write.mode("overwrite").option("header", True).csv(r"C:\Users\sansk\Desktop\Big_Data_Assignmnet\Filtered_Data\original_filtered_30_39_csv")
```
