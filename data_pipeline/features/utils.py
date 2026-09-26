from typing import Dict, Any

def get_val(raw_segment: Dict[str, Any], key: str, default: Any) -> Any:
    val = raw_segment.get(key, default)
    if isinstance(val, dict) and "value" in val:
        v = val["value"]
        if v is None:
            return float('nan') if isinstance(default, (float, int)) else "missing"
        return v
    if val is None:
        return float('nan') if isinstance(default, (float, int)) else "missing"
    return val
