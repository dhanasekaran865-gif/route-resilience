import unittest
from scripts.inspect_graphml import check_routing_attributes

class TestGraphMLInspection(unittest.TestCase):
    def test_check_routing_attributes(self):
        edges_sample = [
            ("1", "2", {"length": "150.5", "highway": "primary", "maxspeed": "30 mph", "oneway": "yes"}),
            ("2", "3", {"geometry": "LINESTRING (-0.125 51.5, -0.124 51.51)"})
        ]
        
        attrs = check_routing_attributes(edges_sample)
        
        self.assertTrue(attrs["length"])
        self.assertTrue(attrs["highway"])
        self.assertTrue(attrs["speed"]) # Because maxspeed is present
        self.assertTrue(attrs["oneway"])
        self.assertTrue(attrs["geometry"])
        
    def test_check_routing_attributes_missing(self):
        edges_sample = [
            ("1", "2", {"other_attr": "value"})
        ]
        
        attrs = check_routing_attributes(edges_sample)
        
        self.assertFalse(attrs["length"])
        self.assertFalse(attrs["highway"])
        self.assertFalse(attrs["speed"])
        self.assertFalse(attrs["oneway"])
        self.assertFalse(attrs["geometry"])

if __name__ == '__main__':
    unittest.main()
