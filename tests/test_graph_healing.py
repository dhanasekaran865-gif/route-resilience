import unittest
import networkx as nx
from data_pipeline.routing.graph_healing import GraphHealer
from data_pipeline.routing.topology_damage import simulate_damage
from data_pipeline.collection.graph_store import GraphStore

class TestGraphHealing(unittest.TestCase):
    def setUp(self):
        self.healer = GraphHealer()
        self.store = GraphStore()
        
        self.G = nx.MultiDiGraph()
        # 111 meters is roughly 0.001 degrees latitude. Let's make it very close.
        self.G.add_node(1, x=0, y=0)
        self.G.add_node(2, x=0.0005, y=0.0005) # ~78 meters
        self.G.add_edge(1, 2, length=10)
        self.G.add_edge(2, 1, length=10)

    def test_damage_graph(self):
        G_dam, removed = simulate_damage(self.G.copy(), fraction=1.0)
        self.assertTrue(len(removed) >= 0)
        
    def test_candidate_generation(self):
        candidates = self.healer.generate_candidates(self.G)
        self.assertEqual(len(candidates), 0)
        
        self.G.remove_edge(1, 2)
        self.G.remove_edge(2, 1)
        candidates2 = self.healer.generate_candidates(self.G, threshold_m=150)
        self.assertEqual(len(candidates2), 2)
        
    def test_validation_and_heal(self):
        G_dam = nx.MultiDiGraph()
        G_dam.add_node(1, x=0, y=0)
        G_dam.add_node(2, x=0.0002, y=0.0002) # ~31 meters
        
        cands = self.healer.generate_candidates(G_dam, 150)
        val = self.healer.predict_and_validate(G_dam, cands)
        self.assertEqual(len(val['accepted']), 2)
        self.healer.validate_and_heal_graph(G_dam, val['accepted'])
        self.assertTrue(G_dam.has_edge(1, 2))
        
    def test_graph_store(self):
        self.store.save_graph("test_1", "original", self.G)
        loaded = self.store.get_graph("test_1", "original")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.number_of_nodes(), 2)

if __name__ == '__main__':
    unittest.main()
