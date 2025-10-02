# -*- coding: utf-8 -*-
"""app.py - Flask Air Quality API with ML Forecast"""

import os
import time
import requests
import json
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from forecast_service import AQIForecastService

class AirQualityAPI:
    def __init__(self):
        # API Keys from .env
        self.airvisual_api_key = os.getenv("AIRVISUAL_API_KEY")
        self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        # API Endpoints
        self.nominatim_url = "https://nominatim.openstreetmap.org/search"
        self.airvisual_url = "http://api.airvisual.com/v2/nearest_city"
        self.openweather_url = "http://api.openweathermap.org/data/2.5/air_pollution"
        self.gemini_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash-lite:generateContent?key={self.gemini_api_key}"
        )

    def get_coordinates(self, city: str) -> Optional[Dict[str, float]]:
        """Get latitude & longitude from city name via Nominatim"""
        try:
            params = {"q": city, "format": "json", "limit": 1}
            headers = {"User-Agent": "AirQualityApp/1.0 (me@example.com)"}
            response = requests.get(self.nominatim_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            if data:
                return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}
            return None
        except requests.RequestException as e:
            print(f"Error getting coordinates: {e}")
            return None

    def get_airvisual_data(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Get AQI and weather data from AirVisual"""
        try:
            params = {"lat": lat, "lon": lon, "key": self.airvisual_api_key}
            response = requests.get(self.airvisual_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error getting AirVisual data: {e}")
            return None

    def get_openweather_pollutants(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Get detailed pollutant data from OpenWeather"""
        try:
            params = {"lat": lat, "lon": lon, "appid": self.openweather_api_key}
            response = requests.get(self.openweather_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error getting OpenWeather pollutant data: {e}")
            return None

    def get_openweather_pollutants_history(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Get historical pollutant data from OpenWeather for the last 6 hours"""
        try:
            end = int(time.time())
            start = end - (6 * 3600)
            params = {
                "lat": lat,
                "lon": lon,
                "start": start,
                "end": end,
                "appid": self.openweather_api_key,
            }
            history_url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
            response = requests.get(history_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error getting OpenWeather pollutant history: {e}")
            return None

    def parse_pollutant_history(self, history_json: Optional[Dict[str, Any]]) -> Dict[str, list]:
        """Parse OpenWeather history JSON into lists per pollutant for the model"""
        hist = { 'pm2_5': [], 'pm10': [], 'co': [], 'no2': [], 'o3': [], 'so2': [] }
        if not history_json or "list" not in history_json:
            return hist
        for entry in history_json["list"]:
            comp = entry.get("components", {})
            hist['pm2_5'].append(comp.get("pm2_5", 0))
            hist['pm10'].append(comp.get("pm10", 0))
            hist['co'].append(comp.get("co", 0))
            hist['no2'].append(comp.get("no2", 0))
            hist['o3'].append(comp.get("o3", 0))
            hist['so2'].append(comp.get("so2", 0))
        return hist

    def get_openweather_pollutants_forecast(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Get forecast pollutant data from OpenWeather (next hours/days)"""
        try:
            params = {"lat": lat, "lon": lon, "appid": self.openweather_api_key}
            forecast_url = "http://api.openweathermap.org/data/2.5/air_pollution/forecast"
            response = requests.get(forecast_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error getting OpenWeather pollutant forecast: {e}")
            return None

    def parse_pollutant_forecast(self, forecast_json: Optional[Dict[str, Any]]) -> list:
        """Parse OpenWeather forecast JSON into a list of entries with dt and components"""
        entries = []
        if not forecast_json or "list" not in forecast_json:
            return entries
        for entry in forecast_json["list"]:
            dt = entry.get("dt")
            comp = entry.get("components", {})
            entries.append({
                "dt": dt,
                "components": {
                    "pm2_5": comp.get("pm2_5", 0),
                    "pm10": comp.get("pm10", 0),
                    "co": comp.get("co", 0),
                    "no2": comp.get("no2", 0),
                    "o3": comp.get("o3", 0),
                    "so2": comp.get("so2", 0),
                }
            })
        return entries

    def categorize_aqi(self, aqi_value: int) -> str:
        """Categorize AQI according to US standards"""
        if aqi_value <= 50:
            return "Good"
        elif aqi_value <= 100:
            return "Moderate"
        elif aqi_value <= 200:
            return "Unhealthy"
        elif aqi_value <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"

    def generate_ai_insights(self, category: str) -> Dict[str, str]:
        """Generate AI insights using Gemini API"""
        try:
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": (
                                    f"Generate a JSON object only with fields 'tip', "
                                    f"'fun_fact', and 'challenge' based on this air quality "
                                    f"category: {category}. "
                                    f"'tip': start with a friendly phrase (like 'Hey' or 'Reminder'), "
                                    f"mention the {category} explicitly, give clear advice, max 12 words. "
                                    f"'fun_fact': write in style 'Did you know? ...', interesting and positive, "
                                    f"1-2 sentences, include 1-2 fun emojis. "
                                    f"'challenge': write as a small daily task the user can do, "
                                    f"actionable and encouraging, include 1 motivating emoji. "
                                    f"Output only JSON, no explanations."
                                )
                            }
                        ]
                    }
                ],
                "generationConfig": {"responseMimeType": "application/json"},
            }

            headers = {"Content-Type": "application/json"}
            response = requests.post(self.gemini_url, json=payload, headers=headers)
            response.raise_for_status()

            data = response.json()
            insight_text = data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(insight_text)

        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            print(f"Error generating AI insights: {e}")
            return {
                "tip": f"Hey! Air quality is {category} today. Stay informed!",
                "fun_fact": "Did you know? Trees can improve air quality! 🌳",
                "challenge": "Take a walk in a green area today! 🚶",
            }

    def process_air_quality_request(self, city: str) -> Dict[str, Any]:
        """Main processing function"""
        print(f"Processing air quality request for: {city}")

        coords = self.get_coordinates(city)
        if not coords:
            return {"success": False, "error": "Could not find coordinates"}

        airvisual_data = self.get_airvisual_data(coords["lat"], coords["lon"])
        if not airvisual_data or "data" not in airvisual_data:
            return {"success": False, "error": "Could not retrieve air quality data"}

        # Ambil pollutant dari OpenWeather
        openweather_data = self.get_openweather_pollutants(coords["lat"], coords["lon"])
        pollutant_data = {}
        openweather_pollutants = {}
        if openweather_data and "list" in openweather_data and len(openweather_data["list"]) > 0:
            components = openweather_data["list"][0].get("components", {})
            pollutant_data = {
                "CO": components.get("co", 0),
                "NO2": components.get("no2", 0),
                "O3": components.get("o3", 0),
                "SO2": components.get("so2", 0),
                "PM2.5": components.get("pm2_5", 0),
                "PM10": components.get("pm10", 0),
            }
            # Format untuk model (pm2_5, pm10, co, no2, o3, so2)
            openweather_pollutants = {
                "pm2_5": components.get("pm2_5", 0),
                "pm10": components.get("pm10", 0),
                "co": components.get("co", 0),
                "no2": components.get("no2", 0),
                "o3": components.get("o3", 0),
                "so2": components.get("so2", 0),
            }

        aqi_value = airvisual_data["data"]["current"]["pollution"]["aqius"]
        category = self.categorize_aqi(aqi_value)
        insights = self.generate_ai_insights(category)

        # Fetch pollutant history and parse
        history_json = self.get_openweather_pollutants_history(coords["lat"], coords["lon"])
        pollutant_history = self.parse_pollutant_history(history_json)

        # Fetch pollutant forecast and parse (for daily usage only)
        forecast_json = self.get_openweather_pollutants_forecast(coords["lat"], coords["lon"])
        pollutant_forecast_entries = self.parse_pollutant_forecast(forecast_json)

        # Get forecast (jika service available)
        if 'forecast_service' in globals() and FORECAST_ENABLED:
            try:
                forecast_data = forecast_service.get_forecast(
                    city=city,
                    current_pollutants=openweather_pollutants,
                    coords=coords,
                    historical_pollutants=pollutant_history,
                    openweather_forecast=pollutant_forecast_entries
                )
            except Exception as e:
                print(f"Forecast generation failed: {e}")
                forecast_data = {"hourly": [], "daily": [], "method": "error"}
        else:
            forecast_data = {"hourly": [], "daily": [], "method": "service_unavailable"}

        response_data = {
            "success": True,
            "city": airvisual_data["data"]["city"],
            "country": airvisual_data["data"]["country"],
            "coords": coords,
            "aqi": aqi_value,
            "category": category,
            "weather": {
                "temp": airvisual_data["data"]["current"]["weather"]["tp"],
                "humidity": airvisual_data["data"]["current"]["weather"]["hu"],
                "wind_ms": airvisual_data["data"]["current"]["weather"]["ws"],
                "wind_kmh": round(airvisual_data["data"]["current"]["weather"]["ws"] * 3.6, 1)
            },
            "insight": insights,
            "forecast": forecast_data,
        }

        if pollutant_data:
            response_data["pollutants"] = pollutant_data

        return response_data


# Flask App Setup
app = Flask(__name__)
CORS(app)
api = AirQualityAPI()

# Initialize forecast service
try:
    forecast_service = AQIForecastService()
    FORECAST_ENABLED = True
    print("ML Forecast service initialized successfully")
except Exception as e:
    FORECAST_ENABLED = False
    print(f"Warning: Forecast service not available - {e}")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "Air Quality API is running",
        "endpoints": {
            "/air-quality": "POST - Get current air quality + hourly & daily forecast"
        },
        "forecast_enabled": FORECAST_ENABLED
    })


@app.route("/air-quality", methods=["POST"])
def air_quality_webhook():
    """Get current air quality + forecast"""
    data = request.get_json()
    city = data.get("city", "")

    if not city:
        return jsonify({"success": False, "error": "City parameter is required"}), 400

    result = api.process_air_quality_request(city)
    if not result.get("success"):
        return jsonify(result), 400

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=8081)
