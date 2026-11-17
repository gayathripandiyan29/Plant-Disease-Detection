# ==========================================
# SMART CROP CARE RECOMMENDATION
# Standalone Feature - Plant Disease Project
# ==========================================

def get_crop_care_recommendation(
    disease_name,
    severity,
    temperature=None,
    humidity=None,
    rainfall=None
):

    disease = disease_name.lower()
    severity = severity.lower()

    recommendations = []

    # --------------------------------------
    # 1. GENERAL CARE BASED ON SEVERITY
    # --------------------------------------

    if severity == "low":
        recommendations.append(
            "Monitor the plant regularly and remove "
            "only clearly affected leaves."
        )

    elif severity == "medium":
        recommendations.append(
            "Inspect the plant carefully and remove "
            "severely affected leaves."
        )

    elif severity == "high":
        recommendations.append(
            "Separate severely affected plants where "
            "practical and seek agricultural expert advice."
        )

    else:
        recommendations.append(
            "Check the plant condition and monitor it regularly."
        )

    # --------------------------------------
    # 2. DISEASE-BASED CARE
    # --------------------------------------

    if "bacterial" in disease:

        recommendations.append(
            "Avoid overhead watering and prevent water "
            "from splashing between plants."
        )

        recommendations.append(
            "Clean gardening tools after handling infected plants."
        )

    elif "early_blight" in disease or "early blight" in disease:

        recommendations.append(
            "Remove infected lower leaves and keep foliage dry."
        )

        recommendations.append(
            "Maintain good air circulation around the plant."
        )

    elif "late_blight" in disease or "late blight" in disease:

        recommendations.append(
            "Inspect nearby plants for rapidly spreading dark patches."
        )

        recommendations.append(
            "Seek local agricultural advice promptly for disease control."
        )

    elif "leaf_mold" in disease or "leaf mold" in disease:

        recommendations.append(
            "Improve ventilation and reduce excessive humidity."
        )

        recommendations.append(
            "Check both upper and lower leaf surfaces regularly."
        )

    elif "septoria" in disease:

        recommendations.append(
            "Remove affected lower leaves and avoid splashing water."
        )

        recommendations.append(
            "Remove infected plant debris from the growing area."
        )

    elif "spider_mite" in disease or "spider mite" in disease:

        recommendations.append(
            "Inspect the underside of leaves for tiny pests and webbing."
        )

        recommendations.append(
            "Use locally approved pest management methods if necessary."
        )

    elif "target_spot" in disease or "target spot" in disease:

        recommendations.append(
            "Remove badly affected leaves and improve air circulation."
        )

        recommendations.append(
            "Avoid excessive moisture on the foliage."
        )

    elif "yellowleaf" in disease or "yellow leaf" in disease:

        recommendations.append(
            "Monitor whiteflies because they can spread this viral disease."
        )

        recommendations.append(
            "Use healthy seedlings and seek local agricultural advice."
        )

    elif "mosaic" in disease:

        recommendations.append(
            "Clean hands and tools to reduce the spread of plant viruses."
        )

        recommendations.append(
            "Remove severely infected plants where appropriate."
        )

    # --------------------------------------
    # 3. WEATHER-BASED CARE
    # --------------------------------------

    if humidity is not None:

        if humidity >= 80:

            recommendations.append(
                "High humidity detected: improve airflow and "
                "avoid keeping leaves wet."
            )

        elif humidity <= 35:

            recommendations.append(
                "Low humidity detected: monitor soil moisture "
                "and water according to crop requirements."
            )

    if temperature is not None:

        if temperature >= 35:

            recommendations.append(
                "High temperature detected: monitor plants for "
                "heat stress and provide suitable shade if needed."
            )

        elif temperature <= 15:

            recommendations.append(
                "Cool conditions detected: monitor plant growth "
                "and protect sensitive plants when necessary."
            )

    if rainfall is not None:

        if rainfall >= 10:

            recommendations.append(
                "Recent rainfall is high: check drainage and "
                "avoid waterlogging."
            )

        elif rainfall == 0:

            recommendations.append(
                "No current rainfall detected: check soil moisture "
                "before watering."
            )

    # --------------------------------------
    # 4. GENERAL RECOMMENDATION
    # --------------------------------------

    recommendations.append(
        "Use balanced fertilizer according to soil testing "
        "and crop requirements. Fertilizer does not cure disease."
    )

    return recommendations


# ==========================================
# TEST THE FEATURE
# ==========================================

if __name__ == "__main__":

    disease_name = "Tomato_Early_blight"
    severity = "Medium"

    temperature = 32
    humidity = 85
    rainfall = 12

    print("\n🌱 SMART CROP CARE RECOMMENDATIONS")
    print("----------------------------------")

    print("Disease:", disease_name)
    print("Severity:", severity)

    care_tips = get_crop_care_recommendation(
        disease_name,
        severity,
        temperature,
        humidity,
        rainfall
    )

    for number, tip in enumerate(care_tips, start=1):
        print(f"\n{number}. {tip}")