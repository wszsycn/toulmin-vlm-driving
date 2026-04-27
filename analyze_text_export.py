"""Export SBERT similarity matrix + diversity stats to JSON for paper figures."""
import json
sim_matrix = {
    "cot_sft":        {"cot_sft": 1.000, "sc_cot_sft_v3": 0.909, "toulmin_ipo_v3": 0.749, "toulmin_kto_v3": 0.751, "toulmin_sft_v3": 0.752},
    "sc_cot_sft_v3":  {"cot_sft": 0.909, "sc_cot_sft_v3": 1.000, "toulmin_ipo_v3": 0.748, "toulmin_kto_v3": 0.752, "toulmin_sft_v3": 0.752},
    "toulmin_ipo_v3": {"cot_sft": 0.749, "sc_cot_sft_v3": 0.748, "toulmin_ipo_v3": 1.000, "toulmin_kto_v3": 0.917, "toulmin_sft_v3": 0.932},
    "toulmin_kto_v3": {"cot_sft": 0.751, "sc_cot_sft_v3": 0.752, "toulmin_ipo_v3": 0.917, "toulmin_kto_v3": 1.000, "toulmin_sft_v3": 0.933},
    "toulmin_sft_v3": {"cot_sft": 0.752, "sc_cot_sft_v3": 0.752, "toulmin_ipo_v3": 0.932, "toulmin_kto_v3": 0.933, "toulmin_sft_v3": 1.000},
}
diversity = {
    "cot_sft":        {"avg_len": 529, "unique_pct": 17.8, "yes_no_centroid_sim": 0.928},
    "sc_cot_sft_v3":  {"avg_len": 549, "unique_pct": 10.8, "yes_no_centroid_sim": 0.935},
    "toulmin_ipo_v3": {"avg_len": 495, "unique_pct": 54.2, "yes_no_centroid_sim": 0.944},
    "toulmin_kto_v3": {"avg_len": 490, "unique_pct": 53.5, "yes_no_centroid_sim": 0.946},
    "toulmin_sft_v3": {"avg_len": 508, "unique_pct": 55.1, "yes_no_centroid_sim": 0.945},
}
json.dump({"similarity_matrix": sim_matrix, "diversity_stats": diversity},
          open("/workspace/PSI_change/json_mode_90/predictions/sbert_analysis.json", "w"),
          indent=2)
print("Saved to predictions/sbert_analysis.json")
