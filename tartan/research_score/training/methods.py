"""Single registry of the four formal transfer methods."""
METHODS={
 "proxy_a":{"init":"nuplan","source":False,"wrapper":False,"profile":"proxy_ab"},
 "proxy_b":{"init":"nuplan","source":False,"wrapper":True,"profile":"proxy_ab"},
 "pretrain_finetune":{"init":"nuplan","source":False,"wrapper":False,"profile":"transfer_primary"},
 "pretrain_adapter":{"init":"nuplan","source":False,"wrapper":True,"profile":"transfer_primary"},
 "joint_train":{"init":"nuplan","source":True,"wrapper":False,"profile":"transfer_primary"},
 "emb_cond_diffusion":{"init":"nuplan","source":True,"wrapper":True,"profile":"transfer_primary"},
}
def method_spec(name,profile):
 if name not in METHODS:raise KeyError(name)
 x=METHODS[name].copy()
 if x["profile"]!=profile:raise ValueError(f"{name} belongs to {x['profile']}, not {profile}")
 return x
