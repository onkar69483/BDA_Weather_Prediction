import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# -----------------------------
# 1. SPARK SESSION
# -----------------------------
def create_spark_session():
    spark = SparkSession.builder \
        .appName("CrossTemporalValidation") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

# -----------------------------
# 2. LOAD DATA FROM MULTIPLE YEARS
# -----------------------------
def load_data_from_years(data_dir, years):
    """Load storm data from specific years"""
    print(f"📡 Loading data for years: {years}")
    
    all_data = []
    for year in years:
        file_pattern = f"StormEvents_details-ftp_v1.0_d{year}_c"
        matching_files = [f for f in os.listdir(data_dir) if f.startswith(file_pattern)]
        
        for file_name in matching_files:
            file_path = os.path.join(data_dir, file_name)
            try:
                df = pd.read_csv(file_path, low_memory=False)
                df['YEAR'] = year  # Add year column
                all_data.append(df)
                print(f"✅ Loaded {len(df)} records from {year}")
            except Exception as e:
                print(f"❌ Error loading {file_name}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"📊 Total loaded: {len(combined_df)} records from {len(years)} years")
        return combined_df
    else:
        return None

# -----------------------------
# 3. PREPROCESS DATA
# -----------------------------
def preprocess_data(df):
    """Preprocess the storm data"""
    print("🧹 Preprocessing data...")
    
    # Define severe events
    severe_events = ['Tornado', 'Hail', 'Thunderstorm Wind', 'Flash Flood', 'Hurricane']
    
    # Create target variable
    df['IS_SEVERE'] = df['EVENT_TYPE'].isin(severe_events).astype(int)
    
    # Handle missing values
    df = df.copy()
    
    # Fill numeric columns
    numeric_cols = ['BEGIN_LAT', 'BEGIN_LON', 'MAGNITUDE']
    for col_name in numeric_cols:
        if col_name in df.columns:
            df[col_name] = df[col_name].fillna(df[col_name].median())
    
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
        if damage_col in df.columns:
            df[damage_col] = df[damage_col].apply(convert_damage)
            df[damage_col] = df[damage_col].fillna(0)
    
    df['DAMAGE_TOTAL'] = df['DAMAGE_PROPERTY'] + df['DAMAGE_CROPS']
    
    # Select only needed columns
    feature_cols = ['BEGIN_LAT', 'BEGIN_LON', 'MAGNITUDE', 'DAMAGE_TOTAL', 'IS_SEVERE', 'YEAR']
    available_cols = [col for col in feature_cols if col in df.columns]
    
    df = df[available_cols].dropna()
    
    print(f"📊 Processed data: {len(df)} records")
    return df, [col for col in available_cols if col not in ['IS_SEVERE', 'YEAR']]

# -----------------------------
# 4. CROSS-TEMPORAL VALIDATION
# -----------------------------
def cross_temporal_validation(spark, data_dir):
    """Test model on different time periods"""
    print("⏰ CROSS-TEMPORAL VALIDATION")
    print("=" * 50)
    
    # Test different training/testing splits
    test_scenarios = [
        {
            'name': 'OLD vs NEW',
            'train_years': list(range(1950, 2011)),  # 1950-2010
            'test_years': list(range(2011, 2021))    # 2011-2020
        },
        {
            'name': 'EVEN vs ODD', 
            'train_years': list(range(1950, 2021, 2)),  # Even years
            'test_years': list(range(1951, 2021, 2))    # Odd years
        },
        {
            'name': 'FIRST HALF vs SECOND HALF',
            'train_years': list(range(1950, 1985)),     # 1950-1984
            'test_years': list(range(1985, 2021))       # 1985-2020
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n🔬 Testing Scenario: {scenario['name']}")
        print(f"   Training years: {scenario['train_years'][0]}-{scenario['train_years'][-1]}")
        print(f"   Testing years: {scenario['test_years'][0]}-{scenario['test_years'][-1]}")
        
        # Load training data
        train_df = load_data_from_years(data_dir, scenario['train_years'])
        if train_df is None or len(train_df) == 0:
            print("❌ No training data found")
            continue
            
        # Load testing data  
        test_df = load_data_from_years(data_dir, scenario['test_years'])
        if test_df is None or len(test_df) == 0:
            print("❌ No testing data found")
            continue
        
        # Preprocess both datasets
        train_processed, feature_cols = preprocess_data(train_df)
        test_processed, _ = preprocess_data(test_df)
        
        if len(train_processed) == 0 or len(test_processed) == 0:
            print("❌ No data after preprocessing")
            continue
        
        # Convert to Spark
        train_spark = spark.createDataFrame(train_processed)
        test_spark = spark.createDataFrame(test_processed)
        
        print(f"📊 Training set: {train_spark.count()} records")
        print(f"📊 Testing set: {test_spark.count()} records")
        
        # Check class distribution
        train_dist = train_spark.groupBy("IS_SEVERE").count().collect()
        test_dist = test_spark.groupBy("IS_SEVERE").count().collect()
        
        print("   Training class distribution:")
        for row in train_dist:
            print(f"     IS_SEVERE={row['IS_SEVERE']}: {row['count']}")
        
        print("   Testing class distribution:")  
        for row in test_dist:
            print(f"     IS_SEVERE={row['IS_SEVERE']}: {row['count']}")
        
        # Train model
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        rf = RandomForestClassifier(featuresCol="features", labelCol="IS_SEVERE", 
                                  numTrees=50, maxDepth=10, seed=42)
        pipeline = Pipeline(stages=[assembler, rf])
        
        print("🔄 Training model...")
        model = pipeline.fit(train_spark)
        
        # Predict on TEST data (different time period)
        predictions = model.transform(test_spark)
        
        # Evaluate
        evaluator = MulticlassClassificationEvaluator(
            labelCol="IS_SEVERE", 
            predictionCol="prediction", 
            metricName="accuracy"
        )
        accuracy = evaluator.evaluate(predictions)
        
        # Calculate F1 score
        f1_evaluator = MulticlassClassificationEvaluator(
            labelCol="IS_SEVERE",
            predictionCol="prediction", 
            metricName="f1"
        )
        f1_score = f1_evaluator.evaluate(predictions)
        
        print(f"🎯 {scenario['name']} Results:")
        print(f"   Accuracy: {accuracy * 100:.2f}%")
        print(f"   F1-Score: {f1_score * 100:.2f}%")
        
        results.append({
            'scenario': scenario['name'],
            'accuracy': accuracy,
            'f1_score': f1_score,
            'train_years': f"{scenario['train_years'][0]}-{scenario['train_years'][-1]}",
            'test_years': f"{scenario['test_years'][0]}-{scenario['test_years'][-1]}",
            'train_size': train_spark.count(),
            'test_size': test_spark.count()
        })
    
    return results

# -----------------------------
# 5. MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    spark = create_spark_session()
    
    try:
        data_dir = "../data"  # Adjust path to your data directory
        
        if not os.path.exists(data_dir):
            print(f"❌ Data directory not found: {data_dir}")
            exit(1)
        
        print("🌩️ STORM PREDICTION - CROSS-TEMPORAL VALIDATION")
        print("Testing model generalization across different time periods")
        print("=" * 60)
        
        # Run cross-temporal validation
        results = cross_temporal_validation(spark, data_dir)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 FINAL RESULTS SUMMARY")
        print("=" * 60)
        
        for result in results:
            print(f"\n{result['scenario']}:")
            print(f"  Train: {result['train_years']} ({result['train_size']} records)")
            print(f"  Test:  {result['test_years']} ({result['test_size']} records)")
            print(f"  Accuracy: {result['accuracy'] * 100:.2f}%")
            print(f"  F1-Score: {result['f1_score'] * 100:.2f}%")
            
            # Interpretation
            if result['accuracy'] > 0.85:
                print("  ✅ Excellent generalization!")
            elif result['accuracy'] > 0.75:
                print("  ⚠️  Good generalization")
            elif result['accuracy'] > 0.65:
                print("  🔶 Moderate generalization") 
            else:
                print("  ❌ Poor generalization")
        
        # Calculate average performance
        if results:
            avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
            avg_f1 = sum(r['f1_score'] for r in results) / len(results)
            print(f"\n📈 AVERAGE ACROSS ALL SCENARIOS:")
            print(f"  Accuracy: {avg_accuracy * 100:.2f}%")
            print(f"  F1-Score: {avg_f1 * 100:.2f}%")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        spark.stop()
