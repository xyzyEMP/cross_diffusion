METHODS={
 "pretrain_finetune":{"init":"nuplan","source":False,"wrapper":False,"profile":"transfer_primary"},
 "pretrain_adapter":{"init":"nuplan","source":False,"wrapper":True,"profile":"transfer_primary"},
 "joint_train":{"init":"nuplan","source":True,"wrapper":False,"profile":"transfer_primary"},
 "emb_cond_diffusion":{"init":"nuplan","source":True,"wrapper":True,"profile":"transfer_primary"},
 "additive_diff_only":{"init":"nuplan","source":True,"wrapper":True,"profile":"proxy_pair_auxiliary"},
 "ours_score_decomp":{"init":"nuplan","source":True,"wrapper":True,"profile":"strict_pair"},
}
def method_spec(name,profile):
 if name not in METHODS:raise KeyError(name)
 x=METHODS[name].copy()
 if x["profile"]!=profile:raise ValueError(f"{name} belongs to {x['profile']}, not {profile}")
 return x
