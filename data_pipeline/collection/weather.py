import abc
import requests
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class WeatherProvider(abc.ABC):
    @abc.abstractmethod
    def get_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        pass

class MockWeatherProvider(WeatherProvider):
    def __init__(self, preset_risk=None, simulate_failure=False):
        self.preset_risk = preset_risk
        self.simulate_failure = simulate_failure

    def get_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        if self.simulate_failure:
            logger.warning("MockWeatherProvider simulating failure.")
            return {
                "weather_risk": {
                    "value": None,
                    "source": "missing",
                    "status": "missing",
                    "confidence": "none"
                }
            }

        risk_val = self.preset_risk if self.preset_risk is not None else 0.15
        
        return {
            "weather_risk": {
                "value": risk_val,
                "source": "weather_api",
                "status": "available",
                "confidence": "high",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "provider": "MockWeatherProvider",
                "raw_value": {"mock": True}
            }
        }

class OpenMeteoProvider(WeatherProvider):
    def __init__(self):
        self.cache = {}

    def get_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        # Simple cache key: location rounded to ~10km, current hour
        hour = datetime.utcnow().hour
        cache_key = (round(lat, 2), round(lon, 2), hour)
        
        if cache_key in self.cache:
            logger.info("Weather cache hit.")
            return self.cache[cache_key]
            
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,precipitation,wind_speed_10m,visibility"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            current = data.get("current", {})
            precipitation = current.get("precipitation", 0.0)
            wind_speed = current.get("wind_speed_10m", 0.0)
            visibility = current.get("visibility", 10000.0) # default good visibility
            
            # Deterministic prototype feature-engineering formula
            risk = 0.0
            
            if precipitation > 10.0:
                risk += 0.6
            elif precipitation > 5.0:
                risk += 0.4
            elif precipitation > 0.0:
                risk += 0.2
                
            if wind_speed > 30.0:
                risk += 0.3
            elif wind_speed > 15.0:
                risk += 0.1
                
            if visibility < 500:
                risk += 0.6
            elif visibility < 2000:
                risk += 0.3
                
            risk = min(risk, 1.0)
            
            result = {
                "weather_risk": {
                    "value": round(risk, 2),
                    "source": "weather_api",
                    "status": "available",
                    "confidence": "medium",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "provider": "Open-Meteo",
                    "raw_value": current
                }
            }
            
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Weather API request failed: {e}")
            return {
                "weather_risk": {
                    "value": None,
                    "source": "missing",
                    "status": "missing",
                    "confidence": "none"
                }
            }
