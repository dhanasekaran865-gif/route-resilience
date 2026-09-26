import unittest
import os
import networkx as nx
from api.graph_routes import demo_healing
from data_pipeline.collection.graph_store import graph_store

class TestHealingVisualization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        res = demo_healing()
        cls.data = res.dict() if hasattr(res, 'dict') else res.model_dump()
        cls.graph_id = cls.data['graph_id']
        cls.G_orig = graph_store.get_graph(cls.graph_id, "original")
        cls.G_damaged = graph_store.get_graph(cls.graph_id, "damaged")
        cls.G_healed = graph_store.get_graph(cls.graph_id, "healed")

    # 1. Original geometry contains the original graph edges
    def test_original_geometry_contains_original_graph_edges(self):
        orig_geoms = self.data.get('original_graph_geometry') or []
        self.assertEqual(len(orig_geoms), self.G_orig.number_of_edges())
        self.assertEqual(len(orig_geoms), 1505)

    # 2. Damaged geometry excludes removed edges
    def test_damaged_geometry_excludes_removed_edges(self):
        damaged_geoms = self.data.get('damaged_graph_geometry') or []
        removed_edges = self.data['damage']['removed_edges']
        expected_count = self.G_orig.number_of_edges() - len(removed_edges)
        self.assertEqual(len(damaged_geoms), expected_count)
        self.assertEqual(len(damaged_geoms), 1453)
        self.assertEqual(self.G_damaged.number_of_edges(), 1453)
        
        # Verify removed edges are not in G_damaged
        for r in removed_edges:
            u, v, k = r['u'], r['v'], r.get('key', 0)
            self.assertFalse(self.G_damaged.has_edge(u, v, k))

    # 3. Removed geometry contains the deliberately removed edges
    def test_removed_geometry_contains_deliberately_removed_edges(self):
        removed_geoms = self.data.get('removed_edge_geometry') or []
        removed_edges = self.data['damage']['removed_edges']
        self.assertEqual(len(removed_geoms), len(removed_edges))
        self.assertEqual(len(removed_geoms), 52)
        for r in removed_geoms:
            self.assertEqual(r['type'], 'LineString')
            self.assertGreaterEqual(len(r['coordinates']), 2)

    # 4. Accepted healed geometry contains exactly accepted candidates
    def test_accepted_healed_geometry_contains_exactly_accepted_candidates(self):
        accepted_geoms = self.data.get('accepted_healed_geometry') or []
        validated_edges = self.data['healing']['validated_edges']
        self.assertEqual(len(accepted_geoms), len(validated_edges))
        self.assertEqual(len(accepted_geoms), 31)
        for a in accepted_geoms:
            self.assertEqual(a['type'], 'LineString')
            self.assertGreaterEqual(len(a['coordinates']), 2)

    # 5. Healed geometry contains surviving damaged edges PLUS accepted candidates
    def test_healed_geometry_contains_surviving_damaged_edges_plus_accepted_candidates(self):
        damaged_count = len(self.data.get('damaged_graph_geometry') or [])
        accepted_count = len(self.data.get('accepted_healed_geometry') or [])
        healed_count = len(self.data.get('healed_graph_geometry') or [])
        
        self.assertEqual(damaged_count, 1453)
        self.assertEqual(accepted_count, 31)
        self.assertEqual(healed_count, damaged_count + accepted_count)
        self.assertEqual(healed_count, 1484)
        self.assertEqual(self.G_healed.number_of_edges(), 1484)

    # 6. Healed geometry count is greater than accepted candidate count
    def test_healed_geometry_count_greater_than_accepted_candidate_count(self):
        healed_count = len(self.data.get('healed_graph_geometry') or [])
        accepted_count = len(self.data.get('accepted_healed_geometry') or [])
        self.assertGreater(healed_count, accepted_count)
        self.assertEqual(healed_count, 1484)
        self.assertEqual(accepted_count, 31)

    # 7. HEALED visualization does not render only accepted candidates
    def test_healed_visualization_does_not_render_only_accepted_candidates(self):
        healed_count = len(self.data.get('healed_graph_geometry') or [])
        accepted_count = len(self.data.get('accepted_healed_geometry') or [])
        # The healed graph must NOT be just the 31 candidates
        self.assertNotEqual(healed_count, accepted_count)
        self.assertEqual(healed_count, 1484)
        self.assertGreater(healed_count, 1000)

    # 8. No fabricated coordinates are introduced
    def test_no_fabricated_coordinates_are_introduced(self):
        all_geoms = (
            (self.data.get('original_graph_geometry') or []) +
            (self.data.get('damaged_graph_geometry') or []) +
            (self.data.get('healed_graph_geometry') or []) +
            (self.data.get('removed_edge_geometry') or []) +
            (self.data.get('accepted_healed_geometry') or [])
        )
        self.assertGreater(len(all_geoms), 4000)
        for geom in all_geoms:
            for coord in geom['coordinates']:
                lon, lat = coord[0], coord[1]
                # Bounding box of London extract: (-0.14, 51.50, -0.11, 51.52)
                self.assertTrue(-0.15 <= lon <= -0.10, f"Longitude out of bounds: {lon}")
                self.assertTrue(51.49 <= lat <= 51.53, f"Latitude out of bounds: {lat}")

    # 9. GeoJSON coordinates are [lon, lat]
    def test_geojson_coordinates_are_lon_lat(self):
        sample_geom = self.data['original_graph_geometry'][0]
        self.assertEqual(sample_geom['type'], 'LineString')
        for coord in sample_geom['coordinates']:
            self.assertEqual(len(coord), 2)
            lon, lat = coord[0], coord[1]
            # Longitude for London is negative (-0.12), Latitude is ~51.51
            self.assertLess(lon, 0.0, "First coordinate must be longitude (negative for London)")
            self.assertGreater(lat, 50.0, "Second coordinate must be latitude (~51.5 for London)")

    # 10. Diagnostics check
    def test_diagnostics_edge_and_geometry_counts(self):
        diag = self.data.get('diagnostics')
        self.assertIsNotNone(diag)
        self.assertEqual(diag['original_edge_count'], 1505)
        self.assertEqual(diag['damaged_edge_count'], 1453)
        self.assertEqual(diag['removed_edge_count'], 52)
        self.assertEqual(diag['accepted_healed_edge_count'], 31)
        self.assertEqual(diag['healed_edge_count'], 1484)
        self.assertEqual(diag['original_geometry_count'], 1505)
        self.assertEqual(diag['damaged_geometry_count'], 1453)
        self.assertEqual(diag['removed_geometry_count'], 52)
        self.assertEqual(diag['accepted_candidate_geometry_count'], 31)
        self.assertEqual(diag['healed_geometry_count'], 1484)

if __name__ == '__main__':
    unittest.main()
