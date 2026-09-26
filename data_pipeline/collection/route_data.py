import abc
import json
import os
from typing import Dict, Any

class DataProvider(abc.ABC):
    @abc.abstractmethod
    def get_route(self, route_id: str) -> Dict[str, Any]:
        pass

class SampleDataProvider(DataProvider):
    def __init__(self, data_path="data/sample/route_data.json"):
        self.data_path = data_path

    def get_route(self, route_id: str) -> Dict[str, Any]:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Sample data file not found: {self.data_path}")
        
        with open(self.data_path, "r") as f:
            data = json.load(f)
            
        if data.get("route_id") == route_id:
            return data
            
        raise ValueError(f"Route ID {route_id} not found in sample data.")
