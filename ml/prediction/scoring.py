from typing import Dict, Any, List

def get_risk_level(score: float) -> str:
    if score <= 30:
        return "LOW"
    elif score <= 60:
        return "MODERATE"
    elif score <= 80:
        return "HIGH"
    else:
        return "VERY HIGH"

def aggregate_route_risk(segment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not segment_results:
        return {
            "overall_risk_score": 0,
            "risk_level": "LOW",
            "average_segment_risk": 0,
            "highest_risk_segment": None,
            "high_risk_segment_count": 0,
            "major_factors": []
        }
        
    scores = [s["risk_score"] for s in segment_results]
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    highest_segment = segment_results[scores.index(max_score)]["segment_id"]
    high_risk_count = sum(1 for s in scores if s > 60)
    
    # Simple factor aggregation: gather top factors from the highest risk segments
    factors_set = set()
    for s in segment_results:
        if s["risk_score"] > 60:
            factors_set.update(s.get("major_factors", []))
            
    # Cap to top 3 factors for brevity
    major_factors = list(factors_set)[:3]
    
    overall_score = round(avg_score * 0.7 + max_score * 0.3)
    
    return {
        "overall_risk_score": int(overall_score),
        "risk_level": get_risk_level(overall_score),
        "average_segment_risk": int(avg_score),
        "highest_risk_segment": highest_segment,
        "high_risk_segment_count": high_risk_count,
        "major_factors": major_factors
    }
