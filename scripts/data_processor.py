import os
import pandas as pd
from pymongo import MongoClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, rand
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline

# -----------------------------
# 1. MINIMAL SPARK CONFIGURATION
# -----------------------------
def create_minimal_spark_session():
    spark = SparkSession.builder \
        .appName("WeatherPredictionBalanced") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "50") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark

# -----------------------------
# 2. BALANCED DATA PROCESSING
# -----------------------------
def load_balanced_data():
    """Load data with better event distribution"""
    print("📡 Loading balanced data from MongoDB...")
    client = MongoClient("mongodb://localhost:27017/")
    db = client["weather_db"]
    collection = db["storm_events"]
    
    # INCREASED LIMIT from 40,000 to 150,000
    cursor = collection.find({}, {
        'EVENT_TYPE': 1,
        'BEGIN_LAT': 1, 
        'BEGIN_LON': 1,
        'MAGNITUDE': 1,
        'DAMAGE_PROPERTY': 1,
        'DAMAGE_CROPS': 1,
        'YEAR': 1  # Added year for better analysis
    }).limit(150000)  # Increased limit
    
    df = pd.DataFrame(list(cursor))
    client.close()
    
    print(f"✅ Loaded {df.shape[0]} records")
    
    # Show year distribution
    if 'YEAR' in df.columns:
        print("📅 Year distribution:")
        print(df['YEAR'].value_counts().sort_index())
    
    return df

def create_balanced_features(spark, pandas_df):
    """Create features with balanced classes"""
    print("🧹 Creating balanced features...")
    
    # Drop MongoDB _id
    if "_id" in pandas_df.columns:
        pandas_df = pandas_df.drop(columns=["_id"])
    
    # Better definition of severe events - focus on truly severe
    truly_severe_events = ['Tornado', 'Hurricane', 'Flash Flood', 'Wildfire']
    moderately_severe = ['Hail', 'Thunderstorm Wind', 'Heavy Rain']
    non_severe = ['Rain', 'Snow', 'Fog', 'Drizzle']
    
    # Create more balanced target
    pandas_df['IS_SEVERE'] = 0  # Default to non-severe
    pandas_df.loc[pandas_df['EVENT_TYPE'].isin(truly_severe_events), 'IS_SEVERE'] = 2  # High severity
    pandas_df.loc[pandas_df['EVENT_TYPE'].isin(moderately_severe), 'IS_SEVERE'] = 1   # Medium severity
    
    # Handle missing values properly (without inplace warnings)
    pandas_df = pandas_df.copy()  # Avoid chained assignment warnings
    
    numeric_cols = ['BEGIN_LAT', 'BEGIN_LON', 'MAGNITUDE']
    for col_name in numeric_cols:
        if col_name in pandas_df.columns:
            pandas_df[col_name] = pandas_df[col_name].fillna(pandas_df[col_name].median())
    
    # Convert damage columns
    def convert_damage(value):
        if pd.isna(value) or value in ["", "0", "0.0", "0K", "0M"]:
            return 0.0
        value = str(value).upper().replace(",", "").strip()
        try:
            if value.endswith("K"):
                return float(value[:-1]) * 1e3
            elif value.endswith("M"):
                return float(value[:-1]) * 1e6
            else:
                return float(value)
        except:
            return 0.0
    
    for damage_col in ['DAMAGE_PROPERTY', 'DAMAGE_CROPS']:
        if damage_col in pandas_df.columns:
            pandas_df[damage_col] = pandas_df[damage_col].apply(convert_damage)
            pandas_df[damage_col] = pandas_df[damage_col].fillna(0)
    
    # Create total damage feature
    pandas_df['DAMAGE_TOTAL'] = pandas_df['DAMAGE_PROPERTY'] + pandas_df['DAMAGE_CROPS']
    
    # Convert to Spark
    spark_df = spark.createDataFrame(pandas_df)
    
    # Filter out null rows
    required_cols = ['BEGIN_LAT', 'BEGIN_LON', 'MAGNITUDE', 'DAMAGE_TOTAL', 'IS_SEVERE']
    for col_name in required_cols:
        if col_name in spark_df.columns:
            spark_df = spark_df.filter(col(col_name).isNotNull())
    
    feature_cols = ['BEGIN_LAT', 'BEGIN_LON', 'MAGNITUDE', 'DAMAGE_TOTAL']
    
    print(f"📊 Final dataset: {spark_df.count()} rows, {len(feature_cols)} features")
    return spark_df, feature_cols

# -----------------------------
# 3. BALANCED MODEL TRAINING
# -----------------------------
def train_balanced_model(df, feature_cols):
    """Train model with balanced evaluation"""
    print("🧠 Training balanced model...")
    
    # Check class distribution
    class_dist = df.groupBy("IS_SEVERE").count().collect()
    print("📊 Class distribution for training:")
    for row in class_dist:
        print(f"  Severity {row['IS_SEVERE']}: {row['count']} records")
    
    # Balance the dataset by undersampling majority class
    severe_counts = {row['IS_SEVERE']: row['count'] for row in class_dist}
    min_count = min(severe_counts.values())
    
    print(f"🔧 Balancing classes to {min_count} samples each...")
    
    # Create balanced dataset by sampling
    balanced_dfs = []
    for severity in severe_counts.keys():
        severity_df = df.filter(col("IS_SEVERE") == severity)
        if severe_counts[severity] > min_count:
            # Undersample if too many
            sample_fraction = min_count / severe_counts[severity]
            severity_df = severity_df.sample(False, sample_fraction, seed=42)
        balanced_dfs.append(severity_df)
    
    balanced_df = balanced_dfs[0]
    for df_part in balanced_dfs[1:]:
        balanced_df = balanced_df.union(df_part)
    
    # Shuffle the balanced dataset
    balanced_df = balanced_df.orderBy(rand())
    
    print(f"📊 Balanced dataset: {balanced_df.count()} rows")
    
    # Single assembler
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    # Random Forest with class weighting
    rf = RandomForestClassifier(
        featuresCol="features", 
        labelCol="IS_SEVERE",
        numTrees=100,
        maxDepth=15,
        seed=42
    )
    
    # Single pipeline
    pipeline = Pipeline(stages=[assembler, rf])
    
    # Train-test split
    train_df, test_df = balanced_df.randomSplit([0.8, 0.2], seed=42)
    
    print("🔄 Training balanced model...")
    model = pipeline.fit(train_df)
    
    # Predictions
    predictions = model.transform(test_df)
    
    # Multiple evaluation metrics
    acc_evaluator = MulticlassClassificationEvaluator(
        labelCol="IS_SEVERE", 
        predictionCol="prediction", 
        metricName="accuracy"
    )
    accuracy = acc_evaluator.evaluate(predictions)
    
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="IS_SEVERE", 
        predictionCol="prediction", 
        metricName="f1"
    )
    f1_score = f1_evaluator.evaluate(predictions)
    
    print(f"✅ Model accuracy: {accuracy * 100:.2f}%")
    print(f"✅ Model F1-score: {f1_score * 100:.2f}%")
    
    # Show prediction distribution
    pred_dist = predictions.groupBy("prediction").count().collect()
    print("📊 Prediction distribution:")
    for row in pred_dist:
        print(f"  Predicted severity {row['prediction']}: {row['count']} records")
    
    return model, predictions, accuracy

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    # Create Spark session
    spark = create_minimal_spark_session()
    
    try:
        # Load and preprocess
        pandas_df = load_balanced_data()
        spark_df, feature_cols = create_balanced_features(spark, pandas_df)
        
        print(f"🔧 Features used: {feature_cols}")
        
        # Train balanced model
        print("\n" + "="*50)
        print("BALANCED MODEL TRAINING")
        print("="*50)
        model, predictions, accuracy = train_balanced_model(spark_df, feature_cols)
        
        # Generate meaningful alerts
        alerts_df = predictions.select(
            "IS_SEVERE", 
            "prediction",
            when(col("prediction") == 2, "🚨 CRITICAL: High severity event")
             .when(col("prediction") == 1, "⚠️  MODERATE: Medium severity event") 
             .otherwise("✅ LOW: Normal conditions")
            .alias("alert_message")
        )
        
        # Show results
        alerts_pd = alerts_df.toPandas()
        print(f"\n📊 Final alert distribution:")
        print(alerts_pd['alert_message'].value_counts())
        
        # Calculate confusion matrix manually
        confusion = alerts_pd.groupby(['IS_SEVERE', 'prediction']).size().unstack(fill_value=0)
        print(f"\n🎯 Confusion Matrix:")
        print(confusion)
        
        # Save to MongoDB
        client = MongoClient("mongodb://localhost:27017/")
        db = client["weather_db"]
        db['balanced_alerts'].delete_many({})
        db['balanced_alerts'].insert_many(alerts_pd.to_dict('records'))
        client.close()
        
        print("✅ Balanced alerts saved to MongoDB")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        spark.stop()
