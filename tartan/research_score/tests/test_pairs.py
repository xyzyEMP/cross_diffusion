from tartan.research_score.data.paired_dataset import validate_pair
def test_proxy_scope():
    p={"pair_level":"proxy","platform_a_embodiment":"diff","platform_b_embodiment":"anymal","supports_claims":{"car_dog_scientific_claim":False}}
    assert validate_pair(p,"proxy_pair_auxiliary")[0]
    p["supports_claims"]["car_dog_scientific_claim"]=True
    assert validate_pair(p,"proxy_pair_auxiliary")== (False,"proxy_claim_forgery")
