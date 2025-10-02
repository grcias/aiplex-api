import pickle
import os
import numpy as np
from datetime import datetime, timedelta

class AQIForecastService:
    """
    Service untuk generate AQI forecast menggunakan trained Random Forest models
    Supports hourly (11 hours) & daily (6 days) forecasts
    """
    POLLUTANT_COLS = ['pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2']

    # City mapping
    CITY_MAP = {
        'jakarta': 'jkt',
        'bogor': 'bgr',
        'bekasi': 'bks',
        'bandung': 'bdg',
        'tangerang': 'tangsel',
        'tangerang selatan': 'tangsel',
        'south tangerang': 'tangsel',
        'depok': 'depok'
    }

    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.load_models()

    def load_models(self):
        """Load all trained Random Forest models"""
        if not os.path.exists(self.models_dir):
            raise FileNotFoundError(f"Models directory '{self.models_dir}' not found!")
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_aqi_model.pkl')]
        if not model_files:
            raise ValueError(f"No trained models found in '{self.models_dir}'!")
        for model_file in model_files:
            city_name = model_file.replace('_aqi_model.pkl', '')
            model_path = os.path.join(self.models_dir, model_file)
            with open(model_path, 'rb') as f:
                self.models[city_name] = pickle.load(f)

    def prepare_features(self, current_pollutants, forecast_time, historical_pollutants=None):
        features = {
            'hour': forecast_time.hour,
            'day_of_week': forecast_time.weekday(),
            'day_of_month': forecast_time.day,
            'month': forecast_time.month,
            'is_weekend': int(forecast_time.weekday() >= 5)
        }

        # pollutant features
        for col in self.POLLUTANT_COLS:
            features[col] = current_pollutants.get(col, 0)

        # lag & rolling features
        for col in self.POLLUTANT_COLS:
            for lag in [1, 3, 6]:
                if historical_pollutants and col in historical_pollutants and len(historical_pollutants[col]) >= lag:
                    features[f'{col}_lag_{lag}h'] = historical_pollutants[col][-lag]
                else:
                    features[f'{col}_lag_{lag}h'] = features[col]

            for window in [3, 6, 12]:
                if historical_pollutants and col in historical_pollutants and len(historical_pollutants[col]) >= window:
                    features[f'{col}_rolling_{window}h'] = np.mean(historical_pollutants[col][-window:])
                elif historical_pollutants and col in historical_pollutants and len(historical_pollutants[col]) > 0:
                    features[f'{col}_rolling_{window}h'] = np.mean(historical_pollutants[col])
                else:
                    features[f'{col}_rolling_{window}h'] = features[col]

        return features

    def predict_aqi(self, city, current_pollutants, forecast_time, historical_pollutants=None):
        city_lower = city.lower()
        if city_lower not in self.models:
            raise ValueError(f"Model for '{city}' not found. Available: {', '.join(self.models.keys())}")
        model_data = self.models[city_lower]
        model = model_data['model']
        feature_cols = model_data['feature_cols']

        features = self.prepare_features(current_pollutants, forecast_time, historical_pollutants)
        X = np.array([features.get(col, 0) for col in feature_cols]).reshape(1, -1)
        pred = max(0, model.predict(X)[0])
        return round(pred, 1)

    def get_aqi_category(self, aqi):
        if aqi <= 50:
            return {'category': 'Good', 'color': '#00e400', 'level': 1, 'description': 'Air quality is satisfactory'}
        elif aqi <= 100:
            return {'category': 'Moderate', 'color': '#ffff00', 'level': 2, 'description': 'Air quality is acceptable'}
        elif aqi <= 150:
            return {'category': 'Unhealthy for Sensitive Groups', 'color': '#ff7e00', 'level': 3,
                    'description': 'Sensitive groups may experience health effects'}
        elif aqi <= 200:
            return {'category': 'Unhealthy', 'color': '#ff0000', 'level': 4,
                    'description': 'Everyone may begin to experience health effects'}
        elif aqi <= 300:
            return {'category': 'Very Unhealthy', 'color': '#8f3f97', 'level': 5,
                    'description': 'Health alert: everyone may experience serious effects'}
        else:
            return {'category': 'Hazardous', 'color': '#7e0023', 'level': 6,
                    'description': 'Health warnings of emergency conditions'}

    def generate_hourly(self, city, current_aqi, current_pollutants, hours=11, historical_pollutants=None):
        forecasts = []
        now = datetime.now()
        if historical_pollutants is None:
            historical_pollutants = {col: [current_pollutants.get(col, 0)] for col in self.POLLUTANT_COLS}

        # Hour 0 pakai AQI existing
        for i in range(hours):
            forecast_time = now + timedelta(hours=i + 1)  # hour 1..11
            pred_aqi = self.predict_aqi(city, current_pollutants, forecast_time, historical_pollutants)
            forecasts.append({
                'timestamp': forecast_time.isoformat(),
                'hour': forecast_time.strftime('%H:%M'),
                'date': forecast_time.strftime('%Y-%m-%d'),
                'aqi': pred_aqi,
                'category': self.get_aqi_category(pred_aqi)
            })
            # update historical
            for col in historical_pollutants:
                historical_pollutants[col].append(current_pollutants.get(col, 0))
                if len(historical_pollutants[col]) > 24:
                    historical_pollutants[col].pop(0)
        return forecasts

    def generate_daily(self, city, current_pollutants, days=6, historical_pollutants=None, openweather_forecast=None):
        forecasts = []
        now = datetime.now()
        if historical_pollutants is None:
            historical_pollutants = {col: [current_pollutants.get(col, 0)] for col in self.POLLUTANT_COLS}

        for day_offset in range(1, days + 1):
            daily_preds = []
            forecast_date = now + timedelta(days=day_offset)
            for hour in [0, 4, 8, 12, 16, 20]:
                forecast_time = forecast_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                # If openweather forecast provided, try to use its components for this hour
                used_pollutants = current_pollutants
                if openweather_forecast:
                    comp = self._get_forecast_components_for_time(openweather_forecast, forecast_time)
                    if comp:
                        used_pollutants = comp
                pred_aqi = self.predict_aqi(city, used_pollutants, forecast_time, historical_pollutants)
                daily_preds.append(pred_aqi)
                # update historical
                for col in historical_pollutants:
                    historical_pollutants[col].append(used_pollutants.get(col, 0))
                    if len(historical_pollutants[col]) > 72:
                        historical_pollutants[col].pop(0)
            forecasts.append({
                'date': forecast_date.strftime('%Y-%m-%d'),
                'day_name': forecast_date.strftime('%A'),
                'aqi_min': round(min(daily_preds), 1),
                'aqi_max': round(max(daily_preds), 1),
                'aqi_avg': round(np.mean(daily_preds), 1),
                'category': self.get_aqi_category(np.mean(daily_preds))
            })
        return forecasts

    def get_full_forecast(self, city, current_aqi, current_pollutants, current_weather=None, historical_pollutants=None, openweather_forecast=None):
        """
        Return dict with 'hourly' and 'daily' forecasts.
        Hourly uses AQI from API as hour 0, next 11 hours predicted.
        """
        return {
            'city': city,
            'generated_at': datetime.now().isoformat(),
            'hourly': self.generate_hourly(city, current_aqi, current_pollutants, hours=11, historical_pollutants=historical_pollutants),
            'daily': self.generate_daily(city, current_pollutants, days=6, historical_pollutants=historical_pollutants, openweather_forecast=openweather_forecast)
        }

    def get_forecast(self, city: str, current_pollutants: dict, coords: dict = None, historical_pollutants: dict = None, openweather_forecast: list = None):
        """
        Compatible method untuk app.py

        Args:
            city: City name (user input)
            current_pollutants: Dict with pm2_5, pm10, co, no2, o3, so2
            coords: {lat, lon} (optional)

        Returns:
            Dict with hourly, daily, method
        """
        city_lower = city.lower()

        # Map city name to model code
        city_code = None
        for name, code in self.CITY_MAP.items():
            if name in city_lower:
                city_code = code
                break

        # Check if model exists
        if not city_code or city_code not in self.models:
            return {
                "hourly": self._generate_fallback_hourly(),
                "daily": self._generate_fallback_daily(),
                "method": "not_available"
            }

        # Validate pollutants
        if not current_pollutants or not any(current_pollutants.values()):
            return {
                "hourly": self._generate_fallback_hourly(),
                "daily": self._generate_fallback_daily(),
                "method": "no_pollutant_data"
            }

        try:
            # Calculate current AQI from pollutants (rough estimate)
            current_aqi = self._estimate_aqi_from_pollutants(current_pollutants)

            # Generate forecast
            forecast = self.get_full_forecast(
                city=city_code,
                current_aqi=current_aqi,
                current_pollutants=current_pollutants,
                historical_pollutants=historical_pollutants,
                openweather_forecast=openweather_forecast
            )

            # Format untuk FE
            return {
                "hourly": [
                    {
                        "time": h["hour"],
                        "aqi": h["aqi"],
                        "category": h["category"]["category"]
                    }
                    for h in forecast["hourly"]
                ],
                "daily": [
                    {
                        "date": d["date"],
                        "day": d["day_name"][:3],  # Mon, Tue, dll
                        "aqi": d["aqi_avg"],
                        "category": d["category"]["category"]
                    }
                    for d in forecast["daily"]
                ],
                "method": "machine_learning"
            }

        except Exception as e:
            print(f"Forecast error for {city}: {e}")
            return {
                "hourly": self._generate_fallback_hourly(),
                "daily": self._generate_fallback_daily(),
                "method": "error"
            }

    def _estimate_aqi_from_pollutants(self, pollutants):
        """Quick AQI estimate dari pollutants (simplified)"""
        # PM2.5 usually dominan
        pm25 = pollutants.get('pm2_5', 0)
        if pm25 <= 12:
            return 50
        elif pm25 <= 35.4:
            return 75
        elif pm25 <= 55.4:
            return 125
        else:
            return 175

    def _generate_fallback_hourly(self):
        """Fallback when forecast not available"""
        from datetime import datetime, timedelta
        now = datetime.now()
        return [
            {
                "time": (now + timedelta(hours=i+1)).strftime("%H:00"),
                "aqi": None,
                "category": None
            }
            for i in range(11)
        ]

    def _generate_fallback_daily(self):
        """Fallback when forecast not available"""
        from datetime import datetime, timedelta
        today = datetime.now()
        return [
            {
                "date": (today + timedelta(days=i+1)).strftime("%Y-%m-%d"),
                "day": (today + timedelta(days=i+1)).strftime("%a"),
                "aqi": None,
                "category": None
            }
            for i in range(6)
        ]

    def _get_forecast_components_for_time(self, openweather_forecast, forecast_time):
        """Find nearest forecast entry (by hour) and return components as model-ready dict"""
        if not openweather_forecast:
            return None
        target_epoch = int(datetime(
            forecast_time.year, forecast_time.month, forecast_time.day,
            forecast_time.hour, 0, 0
        ).timestamp())
        # Find exact match by dt (OpenWeather uses hourly dt seconds)
        for entry in openweather_forecast:
            if entry.get('dt') == target_epoch:
                return entry.get('components', None)
        return None


def get_forecast_from_api_data(city_code, pollutant_data, forecast_service=None, weather_data=None, current_aqi=None):
    """
    Helper untuk Flask app.py
    """
    if forecast_service is None:
        forecast_service = AQIForecastService()

    # Convert pollutant names ke model format
    current_pollutants = {
        'pm2_5': pollutant_data.get('PM2.5', 0),
        'pm10': pollutant_data.get('PM10', 0),
        'co': pollutant_data.get('CO', 0),
        'no2': pollutant_data.get('NO2', 0),
        'o3': pollutant_data.get('O3', 0),
        'so2': pollutant_data.get('SO2', 0)
    }

    if current_aqi is None:
        current_aqi = 50  # fallback moderate

    forecast = forecast_service.get_full_forecast(
        city=city_code,
        current_aqi=current_aqi,
        current_pollutants=current_pollutants,
        current_weather=weather_data
    )

    return forecast
