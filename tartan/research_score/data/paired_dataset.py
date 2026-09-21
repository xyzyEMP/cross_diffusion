VALID_LEVELS={"proxy","strict"}

def validate_pair(pair, profile):
    if pair.get("pair_level") not in VALID_LEVELS:return False,"invalid_pair_level"
    if profile=="proxy_pair_auxiliary":
        if pair.get("platform_a_embodiment")!="diff" or pair.get("platform_b_embodiment")!="anymal":return False,"proxy_platform_mismatch"
        if pair.get("supports_claims",{}).get("car_dog_scientific_claim"):return False,"proxy_claim_forgery"
    return True,"ok"
