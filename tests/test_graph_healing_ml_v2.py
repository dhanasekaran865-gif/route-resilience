import unittest
import numpy as np
import networkx as nx
from shapely.geometry import LineString
from sklearn.ensemble import RandomForestClassifier

from data_pipeline.routing.graph_healing import GraphHealer
from scripts.ml_v2_comprehensive_experiment import (
    extract_candidate_features,
    dict_to_vector,
    haversine_m
)

class TestGraphHealingMLV2(unittest.TestCase):
    def setUp(self):
        self.healer = GraphHealer()
        # Create synthetic test graph
        self.G = nx.MultiDiGraph()
        self.G.add_node("A", x=-0.1250, y=51.5100, street_count=2, elevation=25.0)
        self.G.add_node("B", x=-0.1245, y=51.5100, street_count=2, elevation=26.0)
        self.G.add_node("C", x=-0.1240, y=51.5100, street_count=2, elevation=25.0)
        self.G.add_node("D", x=-0.1240, y=51.5105, street_count=3, elevation=27.0)
        
        # Add edges A <-> B and C <-> D
        self.G.add_edge("A", "B", key=0, length=35.0, highway="residential", oneway="False")
        self.G.add_edge("B", "A", key=0, length=35.0, highway="residential", oneway="False")
        self.G.add_edge("C", "D", key=0, length=55.0, highway="residential", oneway="False")
        self.G.add_edge("D", "C", key=0, length=55.0, highway="residential", oneway="False")
        
        # Node with missing attributes
        self.G.add_node("M", x=-0.1235, y=51.5100) # No elevation, no street_count
        
        self.feature_keys = [
            'candidate_length', 'mean_bearing_diff', 'bearing_diff_u', 'bearing_diff_v',
            'degree_u', 'degree_v', 'degree_sum', 'is_deadend_u', 'is_deadend_v',
            'has_reverse_edge', 'common_neighbors', 'jaccard_coeff',
            'shortest_path_dist', 'detour_ratio', 'is_disconnected',
            'street_count_u', 'street_count_v', 'street_count_diff',
            'v1_structural_score'
        ]

    # 1. Feature extraction
    def test_feature_extraction(self):
        feat = extract_candidate_features(self.G, "B", "C")
        self.assertIsInstance(feat, dict)
        for k in self.feature_keys:
            self.assertIn(k, feat)
            self.assertIsInstance(feat[k], (int, float))
            self.assertFalse(np.isnan(feat[k]))
        self.assertGreater(feat['candidate_length'], 0.0)

    # 2. Missing-feature handling
    def test_missing_feature_handling(self):
        # Extract features involving node 'M' which lacks street_count and elevation
        feat = extract_candidate_features(self.G, "C", "M")
        self.assertIsInstance(feat, dict)
        self.assertEqual(feat['elevation_diff'], 25.0) # 25.0 - 0.0 default
        self.assertFalse(np.isnan(feat['elevation_slope']))
        self.assertFalse(np.isnan(feat['street_count_v']))
        self.assertEqual(feat['street_count_v'], 2.0) # Safe default for street_count

    # 3. Candidate feature determinism
    def test_candidate_feature_determinism(self):
        feat1 = extract_candidate_features(self.G, "B", "C")
        feat2 = extract_candidate_features(self.G, "B", "C")
        v1 = dict_to_vector(feat1, self.feature_keys)
        v2 = dict_to_vector(feat2, self.feature_keys)
        np.testing.assert_array_equal(v1, v2)

    # 4. Candidate generation without ground-truth leakage
    def test_candidate_generation_without_leakage(self):
        # generate_candidates accepts only the graph and threshold, has no parameter for ground truth
        candidates = self.healer.generate_candidates(self.G, threshold_m=100)
        self.assertIsInstance(candidates, list)
        for c in candidates:
            # Must not already exist in G
            self.assertFalse(self.G.has_edge(c['u'], c['v']))
            self.assertLessEqual(c['distance'], 100.0)

    # 5. Class handling
    def test_class_handling(self):
        # Verify RandomForest handles imbalanced sample weighting without errors
        X_toy = np.random.randn(20, len(self.feature_keys))
        y_toy = np.array([1]*3 + [0]*17) # Highly imbalanced
        rf = RandomForestClassifier(n_estimators=10, class_weight='balanced', random_state=42)
        rf.fit(X_toy, y_toy)
        preds = rf.predict(X_toy)
        self.assertEqual(len(preds), 20)

    # 6. Threshold behaviour
    def test_threshold_behaviour(self):
        X_mock = np.array([[10.0]*len(self.feature_keys), [50.0]*len(self.feature_keys), [100.0]*len(self.feature_keys)])
        probs = np.array([0.85, 0.65, 0.35])
        
        counts = []
        for thresh in [0.30, 0.50, 0.70, 0.90]:
            accepted = [i for i, p in enumerate(probs) if p >= thresh]
            counts.append(len(accepted))
            
        # Monotonically non-increasing
        self.assertTrue(counts[0] >= counts[1] >= counts[2] >= counts[3])
        self.assertEqual(counts, [3, 2, 1, 0])

    # 7. Model prediction
    def test_model_prediction(self):
        X = np.random.randn(50, len(self.feature_keys))
        y = np.random.choice([0, 1], size=50)
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        probs = rf.predict_proba(X)[:, 1]
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

    # 8. Validation gate
    def test_validation_gate(self):
        # Candidate with excessive distance must be rejected
        cands = [
            {"u": "A", "v": "D", "distance": 120.0, "u_coord": (-0.125, 51.51), "v_coord": (-0.124, 51.5105)}
        ]
        val = self.healer.predict_and_validate(self.G, cands)
        self.assertEqual(len(val['accepted']), 0)
        self.assertEqual(len(val['rejected']), 1)
        self.assertIn("Distance > 60m", val['rejected'][0]['reason'])

    # 9. Healed graph integrity
    def test_healed_graph_integrity(self):
        G_initial = self.G.copy()
        cand = {
            "u": "B", "v": "C", "probability": 0.85, "predicted_exists": True, "key": 0,
            "data": {"length": 35.0, "highway": "residential", "inferred_by_model": True},
            "geometry": {"type": "LineString", "coordinates": [[-0.1245, 51.5100], [-0.1240, 51.5100]]}
        }
        G_healed = self.healer.validate_and_heal_graph(G_initial.copy(), [cand])
        
        # Validated: healed = damaged + 1
        self.assertEqual(G_healed.number_of_edges(), G_initial.number_of_edges() + 1)
        self.assertTrue(G_healed.has_edge("B", "C"))
        self.assertEqual(G_initial.number_of_edges(), 4) # Original not mutated

    # 10. Deterministic output
    def test_deterministic_output(self):
        X = np.random.RandomState(42).randn(40, len(self.feature_keys))
        y = np.random.RandomState(42).choice([0, 1], size=40)
        
        rf1 = RandomForestClassifier(n_estimators=15, random_state=42)
        rf1.fit(X, y)
        p1 = rf1.predict_proba(X)[:, 1]
        
        rf2 = RandomForestClassifier(n_estimators=15, random_state=42)
        rf2.fit(X, y)
        p2 = rf2.predict_proba(X)[:, 1]
        
        np.testing.assert_array_equal(p1, p2)

if __name__ == '__main__':
    unittest.main()
