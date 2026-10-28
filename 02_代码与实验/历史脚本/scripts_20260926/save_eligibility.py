import json
report = {
    "status": "METHOD_PRECONDITION_NOT_MET",
    "gpu_training_authorized_by_diagnostic": False,
    "source_source_95th": {"spectrum": 0.676778, "color": 0.026574, "illumination": 0.014685},
    "source_test2_distance": {"spectrum": 1.323232, "color": 0.026495, "illumination": 0.014478},
    "above_95th": {"spectrum": True, "color": False, "illumination": False, "count": 1},
    "tast_effect": {
        "rho": 0.375,
        "spectrum_distance_reduction_pct": 39.3,
        "color_distance_reduction_pct": 10.2,
        "mean_clipping_pct": 0.177
    },
    "conditions": {
        "two_families_above_95th": False,
        "tast_reduces_distance_20pct": True,
        "clipping_under_0.5pct": True
    },
    "reason": "Only 1/3 statistical families (spectrum) exceed source-source 95th percentile. Color and illumination do not show significant shift. TAST reduces spectrum distance by 39.3% with low clipping, but precondition requires >=2 families.",
    "note": "Eligibility FAIL_CLOSED. Do not start GPU training."
}
with open("/home/jhy/rare_diag_20260915/taste_20260926/02_eligibility/ELIGIBILITY_RESULT.json", "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
