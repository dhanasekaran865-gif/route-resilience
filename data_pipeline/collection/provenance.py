def make_feature(value, source="available", raw_osm=None, confidence="high"):
    return {
        "value": value,
        "source": source,
        "raw_osm": raw_osm,
        "confidence": confidence
    }
