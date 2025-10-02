import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# AQI Calculator (with Unit Conversion)
# =========================
class AQICalculator:
    """Convert pollutants (µg/m³) to AQI US"""

    # Molecular weights (g/mol) untuk konversi
    MOL_WEIGHTS = {
        "co": 28.01,
        "no2": 46.01,
        "o3": 48.00,
        "so2": 64.07
    }

    # Breakpoints AQI US (sudah dalam unit yang sesuai)
    AQI_BREAKPOINTS = {
"pm2_5": [  # µg/m³
    (0.0, 12.0, 0, 50),
    (12.0, 35.4, 51, 100),  # ← Ubah dari 12.1 jadi 12.0 (overlap OK)
    (35.4, 55.4, 101, 150), # ← Ubah dari 35.5 jadi 35.4
    (55.4, 150.4, 151, 200),
    (150.4, 250.4, 201, 300),
    (250.4, 350.4, 301, 400),
    (350.4, 500.4, 401, 500),
],
"pm10": [  # µg/m³
    (0, 54, 0, 50),
    (54, 154, 51, 100),  # ← Ubah dari 55 jadi 54
    (154, 254, 101, 150),
    (254, 354, 151, 200),
    (354, 424, 201, 300),
    (424, 504, 301, 400),
    (504, 604, 401, 500),
],
        "co": [  # ppm
            (0.0, 4.4, 0, 50),
            (4.5, 9.4, 51, 100),
            (9.5, 12.4, 101, 150),
            (12.5, 15.4, 151, 200),
            (15.5, 30.4, 201, 300),
            (30.5, 40.4, 301, 400),
            (40.5, 50.4, 401, 500),
        ],
        "so2": [  # ppb
            (0, 35, 0, 50),
            (36, 75, 51, 100),
            (76, 185, 101, 150),
            (186, 304, 151, 200),
            (305, 604, 201, 300),
            (605, 804, 301, 400),
            (805, 1004, 401, 500),
        ],
        "no2": [  # ppb
            (0, 53, 0, 50),
            (54, 100, 51, 100),
            (101, 360, 101, 150),
            (361, 649, 151, 200),
            (650, 1249, 201, 300),
            (1250, 1649, 301, 400),
            (1650, 2049, 401, 500),
        ],
        "o3": [  # ppb
            (0, 54, 0, 50),
            (55, 70, 51, 100),
            (71, 85, 101, 150),
            (86, 105, 151, 200),
            (106, 200, 201, 300),
            (201, 300, 301, 400),
            (301, 400, 401, 500),
        ],
    }

    @staticmethod
    def convert_units(concentration, pollutant):
        """
        Convert µg/m³ (dari dataset) ke unit yang sesuai untuk AQI US:
        - PM2.5, PM10: tetap µg/m³
        - CO: convert ke ppm
        - NO2, O3, SO2: convert ke ppb
        """
        if pd.isna(concentration) or concentration < 0:
            return np.nan

        pollutant = pollutant.lower()

        # PM2.5 dan PM10 sudah dalam µg/m³
        if pollutant in ["pm2_5", "pm10"]:
            return concentration

        # CO: µg/m³ → ppm
        # Formula: ppm = (µg/m³ × 24.45) / (molecular_weight × 1000)
        if pollutant == "co":
            ppm = (concentration * 24.45) / (AQICalculator.MOL_WEIGHTS["co"] * 1000)
            return ppm

        # NO2, O3, SO2: µg/m³ → ppb
        # Formula: ppb = (µg/m³ × 24.45) / molecular_weight
        if pollutant in ["no2", "o3", "so2"]:
            mw = AQICalculator.MOL_WEIGHTS[pollutant]
            ppb = (concentration * 24.45) / mw
            return ppb

        return concentration  # fallback

    @staticmethod
    def calculate_aqi(concentration, pollutant):
        """
        Calculate AQI untuk single pollutant
        1. Convert unit dulu (µg/m³ → ppm/ppb)
        2. Match dengan breakpoint
        3. Hitung AQI dengan linear interpolation
        """
        # Step 1: Convert unit
        converted_conc = AQICalculator.convert_units(concentration, pollutant)
        
        if pd.isna(converted_conc) or converted_conc < 0:
            return np.nan

        pollutant = pollutant.lower()
        if pollutant not in AQICalculator.AQI_BREAKPOINTS:
            return np.nan

        # Step 2 & 3: Match breakpoint dan hitung AQI
        for bp_lo, bp_hi, aqi_lo, aqi_hi in AQICalculator.AQI_BREAKPOINTS[pollutant]:
            if bp_lo <= converted_conc <= bp_hi:
                # Linear interpolation
                aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (converted_conc - bp_lo) + aqi_lo
                return round(aqi)

        # Kalau melebihi semua breakpoint, return max AQI
        return 500

    @staticmethod
    def overall_aqi(row):
        """
        Calculate overall AQI = MAX dari semua pollutant AQI
        Sesuai standard US EPA
        """
        pollutants = ["pm2_5", "pm10", "co", "so2", "no2", "o3"]
        aqi_values = []
        aqi_by_pollutant = {}  # Debug: track which pollutant gives max AQI
        
        for pollutant in pollutants:
            if pollutant in row and pd.notna(row[pollutant]):
                aqi = AQICalculator.calculate_aqi(row[pollutant], pollutant)
                if pd.notna(aqi):
                    aqi_values.append(aqi)
                    aqi_by_pollutant[pollutant] = aqi
        
        # Store debug info (which pollutant caused max AQI)
        if aqi_values:
            max_aqi = max(aqi_values)
            # Find pollutant that caused max AQI
            for pol, aqi_val in aqi_by_pollutant.items():
                if aqi_val == max_aqi:
                    # You can store this for debugging
                    break
            return max_aqi
        
        return np.nan


# =========================
# Model Trainer (Random Forest + Prophet Support)
# =========================
class AQIModelTrainer:
    def __init__(self, data_path='data/citylist_cleaned.xlsx', model_type='random_forest'):
        """
        Args:
            data_path: path ke Excel file
            model_type: 'random_forest' atau 'prophet'
        """
        self.data_path = data_path
        self.model_type = model_type.lower()
        self.models = {}
        self.aqi_calc = AQICalculator()
        
        if self.model_type == 'prophet':
            try:
                from prophet import Prophet
                self.Prophet = Prophet
            except ImportError:
                raise ImportError("Prophet not installed. Run: pip install prophet")
    
    def load_and_prepare_data(self):
        """Load data dan calculate AQI target"""
        print("=" * 70)
        print("📂 LOADING DATA FROM EXCEL")
        print("=" * 70)
        
        xls = pd.ExcelFile(self.data_path)
        data_dict = {}
        
        for sheet in xls.sheet_names:
            print(f"\n📄 Loading sheet: {sheet}")
            df = pd.read_excel(self.data_path, sheet_name=sheet)
            
            # Validate columns
            if 'timestamp' not in df.columns:
                print(f"   ⚠️  Skipped: No timestamp column")
                continue
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"   📊 Total records: {len(df)}")
            print(f"   📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            # Calculate US AQI (dengan unit conversion)
            print(f"   🔄 Calculating US AQI (with unit conversion)...")
            
            # Debug: Check raw pollutant values before conversion
            pollutants_check = ['pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2']
            print(f"   📊 Raw pollutant stats (µg/m³):")
            for pol in pollutants_check:
                if pol in df.columns:
                    pol_data = df[pol].dropna()
                    if len(pol_data) > 0:
                        print(f"      {pol.upper():<6}: min={pol_data.min():.2f}, max={pol_data.max():.2f}, mean={pol_data.mean():.2f}")
            
            df['aqi'] = df.apply(self.aqi_calc.overall_aqi, axis=1)
            
            # Remove invalid rows
            before_len = len(df)
            df = df.dropna(subset=['aqi'])
            after_len = len(df)
            
            if before_len > after_len:
                print(f"   ⚠️  Removed {before_len - after_len} rows with invalid AQI")
            
            if df.empty:
                print(f"   ❌ Skipped: No valid data after AQI calculation")
                continue
            
            print(f"   ✅ AQI stats: min={df['aqi'].min():.0f}, max={df['aqi'].max():.0f}, mean={df['aqi'].mean():.1f}")
            
            # Check for extreme outliers (AQI >= 500)
            outliers = df[df['aqi'] >= 500]
            if len(outliers) > 0:
                print(f"   ⚠️  Found {len(outliers)} extreme outliers (AQI=500)")
                
                # Debug: Check which pollutant causes AQI=500
                print(f"   🔍 Analyzing outliers...")
                for idx in outliers.head(3).index:
                    row = outliers.loc[idx]
                    print(f"      Row {idx}: timestamp={row['timestamp']}")
                    for pol in ['pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2']:
                        if pol in row:
                            raw_val = row[pol]
                            converted = self.aqi_calc.convert_units(raw_val, pol)
                            aqi_val = self.aqi_calc.calculate_aqi(raw_val, pol)
                            print(f"         {pol.upper()}: {raw_val:.2f} µg/m³ → {converted:.2f} → AQI {aqi_val}")
                
                # Optional: Cap outliers untuk training stability
                # df.loc[df['aqi'] > 400, 'aqi'] = 400
                # print(f"   ✂️  Capped AQI > 400 to 400")
            
            data_dict[sheet.lower()] = df
        
        return data_dict
    
    def create_features_rf(self, df):
        """Feature engineering untuk Random Forest"""
        df = df.copy()
        
        # Temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Pollutant features (raw µg/m³ values)
        pollutant_cols = ['pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2']
        
        # Lag features (1, 3, 6 hours ago)
        for col in pollutant_cols:
            if col in df.columns:
                df[f'{col}_lag_1h'] = df[col].shift(1)
                df[f'{col}_lag_3h'] = df[col].shift(3)
                df[f'{col}_lag_6h'] = df[col].shift(6)
        
        # Rolling averages (3, 6, 12 hours)
        for col in pollutant_cols:
            if col in df.columns:
                df[f'{col}_rolling_3h'] = df[col].rolling(window=3, min_periods=1).mean()
                df[f'{col}_rolling_6h'] = df[col].rolling(window=6, min_periods=1).mean()
                df[f'{col}_rolling_12h'] = df[col].rolling(window=12, min_periods=1).mean()
        
        # Drop rows dengan NaN dari lag features
        df = df.dropna()
        
        return df
    
    def train_random_forest(self, df, city_name):
        """Train Random Forest model"""
        print(f"\n{'=' * 70}")
        print(f"🌲 TRAINING RANDOM FOREST: {city_name.upper()}")
        print(f"{'=' * 70}")
        
        # Feature engineering
        df_featured = self.create_features_rf(df)
        print(f"📊 Data after feature engineering: {len(df_featured)} samples")
        
        # Define features
        pollutant_cols = ['pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2']
        temporal_cols = ['hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend']
        
        feature_cols = temporal_cols.copy()
        for col in df_featured.columns:
            if any(col.startswith(p) for p in pollutant_cols):
                if col not in feature_cols:
                    feature_cols.append(col)
        
        X = df_featured[feature_cols]
        y = df_featured['aqi']
        
        print(f"📌 Features: {len(feature_cols)} columns")
        print(f"🎯 Target AQI range: {y.min():.0f} - {y.max():.0f} (mean: {y.mean():.1f})")
        
        # Train-test split (80-20, no shuffle untuk time series)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        print(f"\n🔀 Data split:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing:  {len(X_test)} samples")
        
        # Train Random Forest
        print(f"\n🚀 Training Random Forest...")
        model = RandomForestRegressor(
            n_estimators=200,          # More trees untuk stability
            max_depth=20,              # Deeper trees untuk complex patterns
            min_samples_split=10,      # Prevent overfitting
            min_samples_leaf=4,        # Prevent overfitting
            max_features='sqrt',       # Use subset of features per tree
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # MAPE
        train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
        test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
        
        print(f"\n📈 MODEL PERFORMANCE:")
        print(f"   {'Metric':<12} {'Train':<15} {'Test':<15}")
        print(f"   {'-' * 42}")
        print(f"   {'MAE':<12} {train_mae:<15.2f} {test_mae:<15.2f}")
        print(f"   {'RMSE':<12} {train_rmse:<15.2f} {test_rmse:<15.2f}")
        print(f"   {'MAPE (%)':<12} {train_mape:<15.2f} {test_mape:<15.2f}")
        print(f"   {'R²':<12} {train_r2:<15.3f} {test_r2:<15.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 TOP 5 IMPORTANT FEATURES:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"   {row['feature']:<30} {row['importance']:.4f}")
        
        return {
            'model': model,
            'feature_cols': feature_cols,
            'model_type': 'random_forest',
            'metrics': {
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_mape': test_mape,
                'test_r2': test_r2
            }
        }
    
    def train_prophet(self, df, city_name):
        """Train Prophet model (time series forecasting)"""
        print(f"\n{'=' * 70}")
        print(f"📈 TRAINING PROPHET: {city_name.upper()}")
        print(f"{'=' * 70}")
        
        # Prepare data untuk Prophet (needs 'ds' and 'y' columns)
        prophet_df = df[['timestamp', 'aqi']].rename(columns={'timestamp': 'ds', 'aqi': 'y'})
        
        print(f"📊 Total samples: {len(prophet_df)}")
        print(f"🎯 Target AQI range: {prophet_df['y'].min():.0f} - {prophet_df['y'].max():.0f}")
        
        # Train-test split (80-20)
        split_idx = int(len(prophet_df) * 0.8)
        train_df = prophet_df.iloc[:split_idx]
        test_df = prophet_df.iloc[split_idx:]
        
        print(f"\n🔀 Data split:")
        print(f"   Training: {len(train_df)} samples")
        print(f"   Testing:  {len(test_df)} samples")
        
        # Train Prophet
        print(f"\n🚀 Training Prophet...")
        model = self.Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        model.fit(train_df)
        
        # Forecast pada test period
        future = test_df[['ds']]
        forecast = model.predict(future)
        
        y_true = test_df['y'].values
        y_pred = forecast['yhat'].values
        
        # Metrics
        test_mae = mean_absolute_error(y_true, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        test_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        test_r2 = r2_score(y_true, y_pred)
        
        print(f"\n📈 MODEL PERFORMANCE:")
        print(f"   MAE:  {test_mae:.2f}")
        print(f"   RMSE: {test_rmse:.2f}")
        print(f"   MAPE: {test_mape:.2f}%")
        print(f"   R²:   {test_r2:.3f}")
        
        return {
            'model': model,
            'model_type': 'prophet',
            'metrics': {
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_mape': test_mape,
                'test_r2': test_r2
            }
        }
    
    def save_models(self):
        """Save all trained models"""
        print(f"\n{'=' * 70}")
        print("💾 SAVING MODELS")
        print(f"{'=' * 70}")
        
        os.makedirs('models', exist_ok=True)
        
        for city_name, model_data in self.models.items():
            model_path = f'models/{city_name}_aqi_model.pkl'
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"   ✅ {model_path}")
        
        # Save metadata
        metadata = {
            'trained_at': pd.Timestamp.now().isoformat(),
            'model_type': self.model_type,
            'cities': list(self.models.keys()),
            'metrics': {city: data['metrics'] for city, data in self.models.items()}
        }
        
        metadata_path = 'models/training_metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"   ✅ {metadata_path}")
        
        # Summary
        avg_mae = np.mean([m['metrics']['test_mae'] for m in self.models.values()])
        avg_rmse = np.mean([m['metrics']['test_rmse'] for m in self.models.values()])
        avg_mape = np.mean([m['metrics']['test_mape'] for m in self.models.values()])
        
        print(f"\n📊 TRAINING SUMMARY:")
        print(f"   Cities trained: {len(self.models)}")
        print(f"   Model type: {self.model_type.upper()}")
        print(f"   Average Test MAE: {avg_mae:.2f}")
        print(f"   Average Test RMSE: {avg_rmse:.2f}")
        print(f"   Average Test MAPE: {avg_mape:.2f}%")
    
    def train_all(self):
        """Main training pipeline"""
        print("\n" + "=" * 70)
        print("🚀 AQI MODEL TRAINING PIPELINE")
        print("=" * 70)
        print(f"📂 Data source: {self.data_path}")
        print(f"🤖 Model type: {self.model_type.upper()}\n")
        
        # Load data
        data_dict = self.load_and_prepare_data()
        
        if not data_dict:
            print("\n❌ No valid data found! Check your Excel file.")
            return
        
        # Train each city
        for city_name, df in data_dict.items():
            try:
                if self.model_type == 'random_forest':
                    model_data = self.train_random_forest(df, city_name)
                elif self.model_type == 'prophet':
                    model_data = self.train_prophet(df, city_name)
                else:
                    print(f"❌ Unknown model type: {self.model_type}")
                    continue
                
                self.models[city_name] = model_data
                
            except Exception as e:
                print(f"\n❌ ERROR training {city_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save models
        if self.models:
            self.save_models()
            
            print("\n" + "=" * 70)
            print("✨ TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print(f"✅ Models saved in 'models/' directory")
            print(f"✅ Ready to use for forecasting\n")
        else:
            print("\n❌ No models were trained!")


# =========================
# Main
# =========================
if __name__ == "__main__":
    # Pilih model type: 'random_forest' atau 'prophet'
    MODEL_TYPE = 'random_forest'  # Ganti ke 'prophet' kalau mau pake Prophet
    
    print(f"\n🎯 Selected Model: {MODEL_TYPE.upper()}\n")
    
    trainer = AQIModelTrainer(
        data_path='data/citylist_cleaned.xlsx',
        model_type=MODEL_TYPE
    )
    
    trainer.train_all()