def audit(configs):
 keys=("target_manifest","budget","repeat_id","future_points","length_m","controller","eval_manifest")
 ref={k:configs[0][k] for k in keys};differences={c["method"]:{k:c[k] for k in keys if c[k]!=ref[k]} for c in configs}
 return {"passed":not any(differences.values()),"reference":ref,"differences":differences,"method_count":len(configs)}
