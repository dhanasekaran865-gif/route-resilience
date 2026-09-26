import abc
import requests
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TrafficProvider(abc.ABC):
    @abc.abstractmethod
    def get_traffic(self, lat: float, lon: float) -> Dict[str, Any]:
        pass

def map_traffic_level(congestion: float) -> float:
    if congestion < 0.30: return 0.20
    if congestion < 0.60: return 0.50
    if congestion < 0.80: return 0.75
    return 0.95

class MockTrafficProvider(TrafficProvider):
    def __init__(self, current_speed=None, free_flow_speed=None, simulate_failure=False):
        self.current_speed = current_speed if current_speed is not None else 30
        self.free_flow_speed = free_flow_speed if free_flow_speed is not None else 50
        self.simulate_failure = simulate_failure

    def get_traffic(self, lat: float, lon: float) -> Dict[str, Any]:
        if self.simulate_failure:
            logger.warning("MockTrafficProvider simulating failure.")
            return self._fallback_missing()

        return self._build_provenance(self.current_speed, self.free_flow_speed, {"mock": True})
        
    def _fallback_missing(self) -> Dict[str, Any]:
        missing = {"value": None, "source": "missing", "status": "missing", "confidence": "none"}
        return {
            "current_speed": None,
            "free_flow_speed": None,
            "congestion": missing,
            "traffic_level": missing,
            "vehicle_density": missing
        }

    def _build_provenance(self, current: float, free_flow: float, raw_data: Any) -> Dict[str, Any]:
        if free_flow <= 0:
            return self._fallback_missing()
            
        congestion = max(0.0, min(1.0, 1.0 - (current / free_flow)))
        traffic_level = map_traffic_level(congestion)
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        return {
            "current_speed": current,
            "free_flow_speed": free_flow,
            "congestion": {
                "value": round(congestion, 2),
                "source": "traffic_api",
                "status": "derived",
                "confidence": "medium",
                "timestamp": timestamp,
                "provider": "TomTom",
                "derived_from": ["current_speed", "free_flow_speed"],
                "raw_value": raw_data
            },
            "traffic_level": {
                "value": traffic_level,
                "source": "traffic_api",
                "status": "derived",
                "confidence": "medium",
                "timestamp": timestamp,
                "provider": "TomTom",
                "derived_from": ["congestion"],
                "raw_value": raw_data
            },
            "vehicle_density": {
                "value": None,
                "source": "missing",
                "status": "missing",
                "confidence": "none"
            }
        }

class TomTomProvider(TrafficProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.cache = {}

    def get_traffic(self, lat: float, lon: float) -> Dict[str, Any]:
        if not self.api_key:
            logger.warning("TomTomProvider missing API key. Simulating failure.")
            return MockTrafficProvider(simulate_failure=True)._fallback_missing()

        dt = datetime.utcnow()
        time_bucket = int(dt.minute / 5)
        cache_key = (round(lat, 3), round(lon, 3), dt.hour, time_bucket)
        
        if cache_key in self.cache:
            logger.info("Traffic cache hit.")
            return self.cache[cache_key]

        try:
            url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&key={self.api_key}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            flow = data.get("flowSegmentData", {})
            current_speed = flow.get("currentSpeed")
            free_flow_speed = flow.get("freeFlowSpeed")
            
            if current_speed is None or free_flow_speed is None:
                raise ValueError("Missing speed data in TomTom response")
                
            mock = MockTrafficProvider()
            result = mock._build_provenance(current_speed, free_flow_speed, data)
            
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Traffic API request failed: {e}")
            return MockTrafficProvider()._fallback_missing()
