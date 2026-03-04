"""
Registry of representative patient configurations for the Simglucose environment.
Each entry contains the Simglucose identifier, a human-readable label, and a note
on the clinical challenge for that cohort.
"""

PATIENT_CONFIGS = {
    "adolescent#001": {
        "patient_id": "adolescent#001",
        "label": "Adolescent",
        "clinical_note": (
            "Adolescents have highly variable insulin sensitivity driven by puberty hormones, "
            "making glucose control unpredictable and prone to wide swings."
        ),
    },
    "adult#001": {
        "patient_id": "adult#001",
        "label": "Adult",
        "clinical_note": (
            "Adults exhibit the most stable glucose dynamics and predictable insulin response, "
            "serving as the baseline cohort for algorithm benchmarking."
        ),
    },
    "child#001": {
        "patient_id": "child#001",
        "label": "Child",
        "clinical_note": (
            "Children are the most insulin-sensitive and hardest to control; small dose errors "
            "cause large glucose excursions, making hypoglycemia the dominant safety risk."
        ),
    },
}
