import networkx as nx
import sys
import os

def check_routing_attributes(edges_sample):
    attributes = {"length": False, "highway": False, "speed": False, "oneway": False, "geometry": False}
    
    for u, v, data in edges_sample:
        for attr in attributes.keys():
            if attr in data:
                attributes[attr] = True
            # Also check alternative names
            if attr == "speed" and "maxspeed" in data:
                attributes["speed"] = True
    
    return attributes

def inspect_graph(file_path):
    print(f"Loading graph from {file_path}...")
    
    try:
        # Load the graph
        G = nx.read_graphml(file_path)
    except Exception as e:
        print(f"Failed to load graph: {e}")
        return

    print("========================================")
    print("GRAPHML INSPECTION REPORT")
    print("========================================")
    
    # 2. Detect graph type
    graph_type = type(G).__name__
    print(f"Graph Type: {graph_type}")
    
    # 3. Print nodes, edges, attributes
    print(f"Number of Nodes: {G.number_of_nodes()}")
    print(f"Number of Edges: {G.number_of_edges()}")
    
    print("\nGraph Attributes:")
    for k, v in G.graph.items():
        print(f"  {k}: {v}")
        
    print("\nSample Node Attributes (first 3):")
    nodes_iter = iter(G.nodes(data=True))
    for i in range(3):
        try:
            u, data = next(nodes_iter)
            print(f"  Node {u}: {data}")
        except StopIteration:
            break
            
    print("\nSample Edge Attributes (first 3):")
    edges_iter = iter(G.edges(data=True))
    edges_sample = []
    for i in range(3):
        try:
            edge = next(edges_iter)
            edges_sample.append(edge)
            print(f"  Edge {edge[0]} -> {edge[1]}: {edge[2]}")
        except StopIteration:
            break

    # 4 & 5. Check geometry
    has_geom = False
    for u, v, data in edges_sample:
        if 'geometry' in data:
            has_geom = True
            geom = data['geometry']
            print("\nGeometry details (from sample edge):")
            print(f"  Python Type: {type(geom).__name__}")
            print(f"  Sample Value: {geom[:100]}..." if len(str(geom)) > 100 else f"  Sample Value: {geom}")
            
            # Check coordinates format roughly (longitude should be small for London, ~ -0.1)
            # WKT LineString: LINESTRING (-0.123 51.5, ...)
            if "LINESTRING" in str(geom):
                print("  Coordinates appear to be formatted as WKT (Well-Known Text).")
                print("  London longitude is typically ~ -0.1 and latitude ~ 51.5. Check sample to verify order.")
            break
            
    if not has_geom:
        print("\nGeometry: No 'geometry' attribute found in the first 3 edges.")

    # 6. Check CRS
    print("\nCRS Information:")
    crs = G.graph.get('crs', G.graph.get('epsg', 'None found'))
    print(f"  {crs}")

    # 7. Check routing attributes
    print("\nRouting Attributes Present (based on sample):")
    routing_attrs = check_routing_attributes(edges_sample)
    for attr, present in routing_attrs.items():
        print(f"  {attr}: {'Yes' if present else 'No'}")
        
    print("========================================")

if __name__ == "__main__":
    file_path = r"C:\Users\Dhanasekaran\Downloads\united_kingdom-GBR_graphml\london-1912.graphml"
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
    
    inspect_graph(file_path)
