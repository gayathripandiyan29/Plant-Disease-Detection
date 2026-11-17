import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import cv2
import html
from PIL import Image, ImageDraw
from pathlib import Path
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import hashlib
import re
import time

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.txt"

# =========================================================
# 1B. DASHBOARD STYLING (VISUAL CARDS, BARS, DONUT, GAUGE)
# =========================================================
DASHBOARD_CSS = """
<style>
.pd-hero {
    background: linear-gradient(135deg, #e6f7ec 0%, #d3f0dd 55%, #fff6d9 100%);
    border: 1px solid #bfe5cb;
    border-radius: 18px;
    padding: 26px 30px;
    color: #14532d;
    margin-bottom: 18px;
}
.pd-hero h1 { color: #14532d; margin: 0 0 6px 0; font-size: 2.1rem; }
.pd-hero p { color: #2f6b49; margin: 0; font-size: 1rem; }

.pd-result-card {
    background: #f0f8f3;
    border: 1px solid #b7dec7;
    border-radius: 18px;
    padding: 20px 22px;
    color: #1b4332;
}
.pd-result-name { font-size: 1.7rem; font-weight: 800; margin-bottom: 10px; color: #145a38; }
.pd-chip {
    display: inline-block;
    padding: 5px 13px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 700;
    margin: 0 8px 8px 0;
    color: #ffffff;
}
.pd-donut {
    width: 170px; height: 170px; border-radius: 50%;
    margin: 6px auto 10px auto;
    display: flex; align-items: center; justify-content: center;
}
.pd-donut-inner {
    width: 112px; height: 112px; border-radius: 50%;
    background: #ffffff; color: #1b4332;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    font-size: 1.6rem; font-weight: 800; line-height: 1.1;
}
.pd-donut-inner span { font-size: 0.75rem; font-weight: 600; color: #52616B; }
.pd-legend { text-align: center; font-size: 0.9rem; color: #1b4332; font-weight: 600; }

.pd-bar-row { display: flex; align-items: center; gap: 12px; margin: 10px 0; }
.pd-bar-label { flex: 0 0 38%; font-weight: 700; color: #1b4332; font-size: 0.92rem; }
.pd-bar-track { flex: 1; height: 16px; background: #e3efe7; border-radius: 999px; overflow: hidden; }
.pd-bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #2e9e5b, #8fd6a8); }
.pd-bar-value { flex: 0 0 70px; text-align: right; font-weight: 800; color: #145a38; font-size: 0.92rem; }

.pd-card {
    background: #ffffff;
    border: 1px solid #e0e8e3;
    border-radius: 16px;
    padding: 16px 16px 14px 16px;
    min-height: 210px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    margin-bottom: 12px;
}
.pd-card-emoji { font-size: 2.2rem; margin-bottom: 4px; }
.pd-card-title { font-size: 1.05rem; font-weight: 800; color: #1b4332; margin-bottom: 6px; }
.pd-card-text { font-size: 0.88rem; line-height: 1.45; color: #37474f; }

.pd-severity-card {
    display: flex; gap: 22px; align-items: center; flex-wrap: wrap;
    background: #ffffff; border: 1px solid #e0e8e3; border-radius: 16px;
    padding: 18px 20px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    margin-bottom: 14px;
}
.pd-severity-big { flex: 0 0 170px; text-align: center; color: #1b4332; }
.pd-severity-big .num { font-size: 2.4rem; font-weight: 800; line-height: 1.1; }
.pd-severity-big .lbl { font-size: 0.9rem; font-weight: 700; }
.pd-gauge-wrap { flex: 1; min-width: 240px; }
.pd-gauge {
    position: relative; height: 18px; border-radius: 999px;
    background: linear-gradient(90deg, #43a047 0%, #43a047 10%, #fb8c00 10%, #fb8c00 30%, #e53935 30%, #e53935 100%);
    margin: 14px 0 8px 0;
}
.pd-gauge-pointer {
    position: absolute; top: -8px; transform: translateX(-50%);
    width: 6px; height: 34px; border-radius: 4px; background: #1b4332;
    border: 2px solid #ffffff;
}
.pd-gauge-labels { display: flex; justify-content: space-between; font-size: 0.8rem; font-weight: 700; color: #37474f; }

.pd-stat {
    border-radius: 16px; padding: 16px; text-align: center;
    color: #ffffff; margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.10);
}
.pd-stat .emoji { font-size: 2rem; }
.pd-stat .value { font-size: 1.6rem; font-weight: 800; line-height: 1.2; }
.pd-stat .label { font-size: 0.85rem; font-weight: 600; opacity: 0.95; }

.pd-care {
    background: #f6fbf7; border: 1px solid #cfe6d7; border-left: 6px solid #2e9e5b;
    border-radius: 14px; padding: 12px 14px; margin-bottom: 10px; color: #263238;
}
.pd-care .head { font-weight: 800; color: #145a38; margin-bottom: 3px; }
.pd-care .body { font-size: 0.9rem; line-height: 1.4; }

.pd-notes {
    display: flex; gap: 16px; flex-wrap: wrap;
    background: #e8f5e9; border: 1px solid #a5d6a7; border-radius: 16px;
    padding: 16px 18px; margin-top: 8px;
}
.pd-note-item { flex: 1 1 230px; color: #1b4332; font-size: 0.9rem; line-height: 1.4; font-weight: 600; }

.pd-forecast { display: flex; gap: 10px; flex-wrap: wrap; margin: 8px 0 14px 0; }
.pd-day {
    flex: 1 1 105px; min-width: 100px; background: #ffffff;
    border: 1px solid #e0e8e3; border-radius: 14px; padding: 12px 8px;
    text-align: center; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06); color: #1b4332;
}
.pd-day.today { border: 2px solid #2e9e5b; background: #f0f8f3; }
.pd-day .dname { font-weight: 800; font-size: 0.95rem; }
.pd-day .dicon { font-size: 2rem; margin: 2px 0; }
.pd-day .dtemp { font-weight: 800; font-size: 1.05rem; }
.pd-day .dtemp span { color: #6b7c74; font-weight: 600; }
.pd-day .dline { font-size: 0.8rem; color: #37474f; margin-top: 2px; }

.pd-risk { border-radius: 18px; padding: 18px 22px; color: #ffffff; margin: 10px 0 4px 0; }
.pd-risk .rtitle { font-size: 1.6rem; font-weight: 800; }
.pd-risk .rsub { font-size: 0.95rem; opacity: 0.95; margin-top: 4px; }
.pd-gauge.pd-gauge-risk {
    background: linear-gradient(90deg, #43a047 0%, #43a047 33.3%, #fb8c00 33.3%, #fb8c00 66.6%, #e53935 66.6%, #e53935 100%);
}
.pd-reasons { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 12px 0; }
.pd-reason {
    background: #f6fbf7; border: 1px solid #cfe6d7; border-radius: 999px;
    padding: 6px 13px; font-size: 0.85rem; font-weight: 700; color: #1b4332;
}
</style>
"""
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

# =========================================================
# 2. DISEASE INFORMATION
# =========================================================
DISEASE_INFO = {
    "Mango_Anthracnose": {
        "name": "Mango - Anthracnose",
        "description": "A fungal disease that affects mango leaves, flowers and fruit, especially in warm, humid or rainy weather.",
        "symptoms": "Dark brown to black spots or patches on leaves that may enlarge and merge; flowers may dry and drop, and fruit can develop dark sunken spots.",
        "treatment": "Prune and remove badly affected leaves and twigs. Ask a local agriculture expert about approved fungicides and the right spray timing.",
        "precautions": "Avoid working among wet plants and do not leave infected leaves lying under the tree.",
        "prevention": "Keep the canopy open with regular pruning, collect fallen infected leaves and fruit, and keep the orchard clean.",
        "fertilizer": "Use balanced fertilizer and organic manure based on soil testing; fertilizer does not cure the disease."
    },
    "Mango_Bacterial_Canker": {
        "name": "Mango - Bacterial Canker",
        "description": "A bacterial disease that causes spots and cankers on mango leaves, twigs and fruit.",
        "symptoms": "Small water-soaked angular spots on leaves that turn dark and raised; cracks or oozing patches may appear on twigs and fruit.",
        "treatment": "Remove and destroy badly affected leaves and twigs. Ask a local agriculture expert about suitable bacterial disease control.",
        "precautions": "Disinfect pruning tools between trees and avoid pruning in wet weather.",
        "prevention": "Use healthy planting material, protect trees from wounds and strong wind, and keep the orchard clean.",
        "fertilizer": "Maintain balanced nutrition based on soil testing; fertilizer does not cure the disease."
    },
    "Mango_Cutting_Weevil": {
        "name": "Mango - Cutting Weevil",
        "description": "An insect pest whose adults cut and damage the tender young leaves of mango trees.",
        "symptoms": "Young tender leaves look cut or rolled, leaf tips hang or fall off, and new growth looks damaged.",
        "treatment": "Collect and destroy damaged young leaves. Ask a local agriculture expert about approved pest control.",
        "precautions": "Check new shoots regularly, especially after rain when fresh growth appears.",
        "prevention": "Keep the orchard clean, remove fallen damaged leaves, and watch new flushes closely.",
        "fertilizer": "Balanced nutrition helps the tree recover; fertilizer does not control insects."
    },
    "Mango_Die_Back": {
        "name": "Mango - Die Back",
        "description": "A disease in which mango branches and twigs dry from the tip downward, often linked to fungal infection.",
        "symptoms": "Twigs dry from the tip and turn brown or grey, dried leaves stay attached, and the bark may darken or crack.",
        "treatment": "Prune affected branches well below the dried portion and destroy them. Ask a local agriculture expert about wound protection and approved fungicides.",
        "precautions": "Disinfect pruning tools and avoid wounding the tree unnecessarily.",
        "prevention": "Prune regularly in dry weather, keep the tree healthy with proper watering, and avoid water stress.",
        "fertilizer": "Use balanced fertilizer and organic manure based on soil testing; fertilizer does not cure the disease."
    },
    "Mango_Gall_Midge": {
        "name": "Mango - Gall Midge",
        "description": "Tiny insect pests whose larvae cause small galls or bumps on mango leaves and flower buds.",
        "symptoms": "Small raised bumps or wart-like swellings on leaves, leaf curling, and damaged flower buds.",
        "treatment": "Remove and destroy badly affected leaves and shoots. Ask a local agriculture expert about suitable pest management.",
        "precautions": "Check new flushes and flower buds regularly.",
        "prevention": "Keep the orchard clean, monitor early, and follow local advice on soil and canopy care.",
        "fertilizer": "Balanced nutrition supports the tree; fertilizer does not control insects."
    },
    "Mango_Healthy": {
        "name": "Mango - Healthy",
        "description": "No disease pattern was detected in the uploaded leaf image.",
        "symptoms": "No clear disease symptoms detected by the model.",
        "treatment": "No disease treatment is indicated from this prediction.",
        "precautions": "Continue regular inspection and avoid water stress.",
        "prevention": "Prune for good airflow, keep the orchard clean, and monitor new growth.",
        "fertilizer": "Use balanced fertilizer and organic manure according to soil testing and tree age."
    },
    "Mango_Powdery_Mildew": {
        "name": "Mango - Powdery Mildew",
        "description": "A fungal disease that forms a white powdery coating on mango flowers, young leaves and small fruits.",
        "symptoms": "White to grey powdery growth on flower clusters and young leaves; flowers and tiny fruits may drop.",
        "treatment": "Remove severely affected flower clusters or shoots. Ask a local agriculture expert about approved fungicides and spray timing.",
        "precautions": "Watch the tree closely during flowering, especially in humid weather.",
        "prevention": "Prune to open the canopy, keep good airflow, and monitor from the start of flowering.",
        "fertilizer": "Use balanced fertilizer based on soil testing; fertilizer does not cure the disease."
    },
    "Mango_Sooty_Mould": {
        "name": "Mango - Sooty Mould",
        "description": "A black fungal coating that grows on the sticky honeydew left by sap-sucking hoppers and scale insects.",
        "symptoms": "A black sooty layer on leaves that can be rubbed off, sticky leaves, and small sap-sucking insects on shoots.",
        "treatment": "Control the sap-sucking insects behind the honeydew with locally approved methods; the black layer then dries and falls. Ask a local agriculture expert for advice.",
        "precautions": "Inspect leaf undersides and young shoots for hoppers or scale insects.",
        "prevention": "Prune to open the canopy, monitor early for sap-sucking insects, and keep the orchard clean.",
        "fertilizer": "Use balanced fertilizer based on soil testing; fertilizer does not remove sooty mould."
    },
    "Rice_Bacterial_Leaf_Blight": {
        "name": "Rice - Bacterial Leaf Blight",
        "description": "A bacterial disease that dries rice leaves and can reduce yield, often spreading in wet and windy weather.",
        "symptoms": "Water-soaked streaks along the leaf edges that turn yellow, then grey-white, and dry from the tip downward.",
        "treatment": "Seek local agriculture advice quickly. Drain excess water from the field where possible and avoid extra nitrogen.",
        "precautions": "Avoid walking through wet fields and do not move water from infected fields to healthy ones.",
        "prevention": "Use healthy certified seed and tolerant varieties, keep fields and bunds free of weeds, and avoid excess nitrogen.",
        "fertilizer": "Avoid excess nitrogen; use balanced fertilizer based on soil testing."
    },
    "Rice_Brown_Spot": {
        "name": "Rice - Brown Spot",
        "description": "A fungal disease linked to poor soil nutrition and plant stress, causing brown spots on rice leaves and grains.",
        "symptoms": "Oval brown spots with grey or whitish centres on leaves; spots may merge and the leaf can dry.",
        "treatment": "Correct nutrient shortages and ask a local agriculture expert about approved fungicides if the infection is severe.",
        "precautions": "Avoid water stress and use clean seed.",
        "prevention": "Use healthy seed, keep the field free of weeds, and maintain good soil fertility.",
        "fertilizer": "Brown spot is often linked to poor soil fertility; apply balanced nutrients according to soil testing."
    },
    "Rice_Hispa": {
        "name": "Rice - Hispa",
        "description": "An insect pest whose adults and larvae feed on rice leaves.",
        "symptoms": "White parallel streaks or patches where the green leaf tissue has been scraped away, and dry leaf tips.",
        "treatment": "Remove and destroy affected leaves in small fields. Ask a local agriculture expert about pest management.",
        "precautions": "Inspect fields regularly, especially young plants.",
        "prevention": "Avoid very dense planting and excess nitrogen, and keep the field and bunds clean.",
        "fertilizer": "Use balanced fertilizer; fertilizer does not control insects."
    },
    "Rice_Healthy": {
        "name": "Rice - Healthy",
        "description": "No disease pattern was detected in the uploaded leaf image.",
        "symptoms": "No clear disease symptoms detected by the model.",
        "treatment": "No disease treatment is indicated from this prediction.",
        "precautions": "Continue regular field inspection and avoid water stress.",
        "prevention": "Use healthy seed, keep the field and bunds clean, and manage water properly.",
        "fertilizer": "Use balanced fertilizer according to soil testing and crop stage."
    },
    "Rice_Leaf_Blast": {
        "name": "Rice - Leaf Blast",
        "description": "A serious fungal disease of rice that can affect leaves, nodes and panicles, favoured by high humidity and cool nights.",
        "symptoms": "Spindle-shaped spots with grey centres and brown edges on leaves; many spots may merge and dry the leaf.",
        "treatment": "Contact a local agriculture expert promptly about approved fungicides and the right timing.",
        "precautions": "Avoid too much nitrogen and check nearby plants for similar spots.",
        "prevention": "Use resistant varieties and healthy seed, and avoid heavy nitrogen application.",
        "fertilizer": "Avoid excess nitrogen; use balanced fertilizer based on soil testing."
    },
    "Rice_Leaf_Smut": {
        "name": "Rice - Leaf Smut",
        "description": "A fungal disease that causes small black spots on rice leaves and is usually minor.",
        "symptoms": "Small raised black angular spots on the leaf, mainly on older leaves; leaf tips may dry.",
        "treatment": "Usually not severe. Remove badly affected leaves where practical and seek local advice if it spreads.",
        "precautions": "Avoid excess nitrogen and very dense planting.",
        "prevention": "Use healthy seed, balanced fertilizer, and keep the field clean.",
        "fertilizer": "Use balanced fertilizer based on soil testing; avoid excess nitrogen."
    },
    "Rice_Sheath_Blight": {
        "name": "Rice - Sheath Blight",
        "description": "A fungal disease of the rice leaf sheath and leaves, favoured by dense planting and high humidity.",
        "symptoms": "Oval greenish-grey spots on the sheath near the water line that enlarge and merge; plants may fall over in severe cases.",
        "treatment": "Ask a local agriculture expert about approved fungicides and reduce plant density where practical.",
        "precautions": "Avoid excess nitrogen and keep bunds free of weeds.",
        "prevention": "Use proper plant spacing, avoid excess nitrogen, and manage field water.",
        "fertilizer": "Avoid excess nitrogen; use balanced fertilizer based on soil testing."
    },
    "Rice_Tungro": {
        "name": "Rice - Tungro",
        "description": "A viral disease of rice spread by green leafhoppers.",
        "symptoms": "Yellow to orange leaf colour, stunted plants, and fewer tillers.",
        "treatment": "There is no cure for infected plants; seek local advice on removing infected plants and managing leafhoppers.",
        "precautions": "Do not transplant seedlings from infected fields and watch for leafhoppers.",
        "prevention": "Use tolerant varieties, plant at the recommended time, and manage leafhoppers using approved methods.",
        "fertilizer": "Balanced nutrition supports growth but cannot cure a viral infection."
    },
    "Tomato_Bacterial_spot": {
        "name": "Tomato - Bacterial Spot",
        "description": "A bacterial disease that produces dark spots on tomato leaves, stems, and fruit.",
        "symptoms": "Small dark or water-soaked spots, sometimes with yellow margins.",
        "treatment": "Remove severely affected leaves and seek advice on locally approved control methods.",
        "precautions": "Avoid handling wet plants and reduce water splash.",
        "prevention": "Use clean seed, rotate crops, and disinfect tools.",
        "fertilizer": "Maintain balanced nutrition based on soil requirements."
    },
    "Tomato_Early_blight": {
        "name": "Tomato - Early Blight",
        "description": "A fungal disease that often begins on older leaves.",
        "symptoms": "Brown lesions with concentric rings, yellowing, and lower-leaf drop.",
        "treatment": "Remove affected lower leaves and seek local advice on approved fungicides.",
        "precautions": "Keep foliage dry and remove crop debris.",
        "prevention": "Rotate crops, mulch soil, and provide good air circulation.",
        "fertilizer": "Use balanced fertilizer and avoid excessive nitrogen."
    },
    "Tomato_Late_blight": {
        "name": "Tomato - Late Blight",
        "description": "A fast-spreading disease favored by cool and humid weather.",
        "symptoms": "Irregular dark water-soaked patches on leaves and stems; fruit may develop firm brown lesions.",
        "treatment": "Remove affected material safely and seek prompt agricultural guidance.",
        "precautions": "Avoid overhead watering and inspect nearby plants.",
        "prevention": "Use healthy seedlings, maintain spacing, and remove infected debris.",
        "fertilizer": "Maintain balanced nutrition; fertilizer does not cure the disease."
    },
    "Tomato_Leaf_Mold": {
        "name": "Tomato - Leaf Mold",
        "description": "A fungal disease commonly associated with high humidity and poor ventilation.",
        "symptoms": "Pale green or yellow patches on upper leaf surfaces and olive-green mold beneath.",
        "treatment": "Remove severely affected leaves and improve ventilation.",
        "precautions": "Avoid excessive humidity and prolonged leaf wetness.",
        "prevention": "Provide spacing, airflow, and careful irrigation.",
        "fertilizer": "Use balanced fertilizer according to soil and crop needs."
    },
    "Tomato_Septoria_leaf_spot": {
        "name": "Tomato - Septoria Leaf Spot",
        "description": "A fungal disease that causes numerous small spots, often on lower leaves.",
        "symptoms": "Small circular spots with pale centers and dark margins; leaves may yellow and drop.",
        "treatment": "Remove affected lower leaves and seek advice on approved control methods.",
        "precautions": "Avoid splashing water between plants.",
        "prevention": "Rotate crops, mulch soil, and remove infected debris.",
        "fertilizer": "Maintain balanced nutrition based on soil testing."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": "Tomato - Two-Spotted Spider Mites",
        "description": "Tiny sap-feeding pests that can cause stippling and bronzing of leaves.",
        "symptoms": "Fine pale speckles, yellowing, bronzing, and sometimes fine webbing under leaves.",
        "treatment": "Inspect plants carefully and use locally approved integrated pest-management methods.",
        "precautions": "Check leaf undersides and avoid unnecessary broad-spectrum pesticide use.",
        "prevention": "Monitor regularly and conserve beneficial insects where possible.",
        "fertilizer": "Maintain balanced nutrition; fertilizer does not control mites."
    },
    "Tomato__Target_Spot": {
        "name": "Tomato - Target Spot",
        "description": "A fungal disease that produces brown lesions on leaves and may affect stems and fruit.",
        "symptoms": "Brown circular lesions, sometimes with concentric rings, that may enlarge and merge.",
        "treatment": "Remove badly affected leaves and seek local agricultural advice.",
        "precautions": "Avoid excessive moisture on foliage.",
        "prevention": "Improve air circulation, rotate crops, and remove crop residue.",
        "fertilizer": "Use balanced fertilizer according to soil needs."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "name": "Tomato - Yellow Leaf Curl Virus",
        "description": "A viral disease commonly spread by whiteflies.",
        "symptoms": "Upward leaf curling, yellowing, stunted growth, and reduced fruit production.",
        "treatment": "There is no curative treatment for infected plants; seek local advice on removing severely infected plants.",
        "precautions": "Monitor whiteflies and avoid moving infested seedlings.",
        "prevention": "Use healthy seedlings and manage whitefly populations using approved methods.",
        "fertilizer": "Balanced nutrition supports plant health but cannot cure a viral infection."
    },
    "Tomato__Tomato_mosaic_virus": {
        "name": "Tomato - Tomato Mosaic Virus",
        "description": "A viral disease that can cause mottled leaf patterns and reduced plant growth.",
        "symptoms": "Mosaic light and dark green patterns, leaf distortion, and stunted growth.",
        "treatment": "No curative treatment is available; remove severely infected plants where appropriate.",
        "precautions": "Clean hands and tools after handling plants.",
        "prevention": "Use virus-free seed or seedlings and maintain good sanitation.",
        "fertilizer": "Use balanced fertilizer; it does not cure viral disease."
    },
    "Tomato_healthy": {
        "name": "Tomato - Healthy",
        "description": "No disease pattern was detected in the uploaded leaf image.",
        "symptoms": "No clear disease symptoms detected by the model.",
        "treatment": "No disease treatment is indicated from this prediction.",
        "precautions": "Continue regular inspection and avoid overwatering.",
        "prevention": "Maintain good spacing, sanitation, and suitable watering.",
        "fertilizer": "Use balanced fertilizer according to soil testing and crop stage."
    }
}

HEALTHY_CLASSES = {
    "Mango_Healthy",
    "Rice_Healthy",
    "Tomato_healthy"
}

# Confidence levels used for the trust badge and the "retake photo" warning.
HIGH_CONFIDENCE = 85.0
LOW_CONFIDENCE = 60.0
# Below this the AI treats the leaf as "not recognized" and shows no disease name.
UNKNOWN_CONFIDENCE = 50.0

NOTES_HTML = (
    '<div class="pd-notes">'
    '<div class="pd-note-item">🤖 This is an AI-based tool and it may not give a 100% accurate result every time.</div>'
    '<div class="pd-note-item">👩‍🌾 Please consult a local agriculture expert to confirm the disease before taking major action.</div>'
    '<div class="pd-note-item">🌾 Before using any fertilizer or pesticide, get proper advice based on your crop and soil condition.</div>'
    '</div>'
)

# =========================================================
# 3. SMART CROP CARE RECOMMENDATION FUNCTION
# =========================================================
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

    if severity == "low":
        recommendations.append(
            "Monitor the plant regularly and remove only clearly affected leaves."
        )
    elif severity == "medium":
        recommendations.append(
            "Inspect the plant carefully and remove severely affected leaves."
        )
    elif severity == "high":
        recommendations.append(
            "Separate severely affected plants where practical and seek agricultural expert advice."
        )
    else:
        recommendations.append("Check the plant condition and monitor it regularly.")

    if "bacterial" in disease:
        recommendations.extend([
            "Avoid overhead watering and prevent water from splashing between plants.",
            "Clean gardening tools after handling infected plants."
        ])
    elif "early_blight" in disease or "early blight" in disease:
        recommendations.extend([
            "Remove infected lower leaves and keep foliage dry.",
            "Maintain good air circulation around the plant."
        ])
    elif "late_blight" in disease or "late blight" in disease:
        recommendations.extend([
            "Inspect nearby plants for rapidly spreading dark patches.",
            "Seek local agricultural advice promptly for disease control."
        ])
    elif "leaf_mold" in disease or "leaf mold" in disease:
        recommendations.extend([
            "Improve ventilation and reduce excessive humidity.",
            "Check both upper and lower leaf surfaces regularly."
        ])
    elif "septoria" in disease:
        recommendations.extend([
            "Remove affected lower leaves and avoid splashing water.",
            "Remove infected plant debris from the growing area."
        ])
    elif "spider_mite" in disease or "spider mite" in disease:
        recommendations.extend([
            "Inspect the underside of leaves for tiny pests and webbing.",
            "Use locally approved pest management methods if necessary."
        ])
    elif "target_spot" in disease or "target spot" in disease:
        recommendations.extend([
            "Remove badly affected leaves and improve air circulation.",
            "Avoid excessive moisture on the foliage."
        ])
    elif "yellowleaf" in disease or "yellow leaf" in disease:
        recommendations.extend([
            "Monitor whiteflies because they can spread this viral disease.",
            "Use healthy seedlings and seek local agricultural advice."
        ])
    elif "mosaic" in disease:
        recommendations.extend([
            "Clean hands and tools to reduce the spread of plant viruses.",
            "Remove severely infected plants where appropriate."
        ])
    elif "anthracnose" in disease or "die_back" in disease or "die back" in disease:
        recommendations.extend([
            "Prune and destroy affected leaves or twigs, and keep the canopy open.",
            "Collect fallen infected leaves and fruit from under the tree."
        ])
    elif "powdery" in disease or "sooty" in disease:
        recommendations.extend([
            "Check new shoots and flower clusters regularly, especially in humid weather.",
            "Control sap-sucking insects and keep the orchard clean."
        ])
    elif "weevil" in disease or "midge" in disease or "hispa" in disease:
        recommendations.extend([
            "Collect and destroy damaged leaves or shoots.",
            "Inspect new growth regularly and use locally approved pest management if necessary."
        ])
    elif "blast" in disease or "sheath" in disease or "brown_spot" in disease or "brown spot" in disease or "smut" in disease:
        recommendations.extend([
            "Avoid excess nitrogen and keep the field and bunds free of weeds.",
            "Manage field water properly and seek local advice on approved fungicides."
        ])
    elif "tungro" in disease:
        recommendations.extend([
            "Watch for green leafhoppers because they spread this viral disease.",
            "Remove infected plants where appropriate and use tolerant varieties next season."
        ])

    if humidity is not None:
        if humidity >= 80:
            recommendations.append(
                "High humidity detected: improve airflow and avoid keeping leaves wet."
            )
        elif humidity <= 35:
            recommendations.append(
                "Low humidity detected: monitor soil moisture and water according to crop requirements."
            )

    if temperature is not None:
        if temperature >= 35:
            recommendations.append(
                "High temperature detected: monitor plants for heat stress and provide suitable shade if needed."
            )
        elif temperature <= 15:
            recommendations.append(
                "Cool conditions detected: monitor plant growth and protect sensitive plants when necessary."
            )

    if rainfall is not None:
        if rainfall >= 10:
            recommendations.append(
                "Recent/current rainfall is high: check drainage and avoid waterlogging."
            )
        elif rainfall == 0:
            recommendations.append(
                "No current rainfall detected: check soil moisture before watering."
            )

    recommendations.append(
        "Use balanced fertilizer according to soil testing and crop requirements. Fertilizer does not cure disease."
    )
    return recommendations

# =========================================================
# 4. LOAD MODEL AND CLASS NAMES
# =========================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    # The model is trained together with class_names.json, so it is read first.
    # class_names.txt is used only if the json file is missing.
    json_path = BASE_DIR / "models" / "class_names.json"
    if json_path.exists():
        try:
            import json as _json
            data = _json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # {"0": "name"} or {"name": 0}
                if all(str(k).isdigit() for k in data):
                    data = [data[k] for k in sorted(data, key=lambda x: int(x))]
                else:
                    data = [k for k, _ in sorted(data.items(), key=lambda kv: kv[1])]
            names = [str(n).strip() for n in data if str(n).strip()]
            if names:
                return names
        except Exception:
            pass
    if not CLASS_NAMES_PATH.exists():
        return []
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]

model = load_model()
class_names = load_class_names()


def _align_disease_info_with_classes():
    """Match every class name of the trained model with its DISEASE_INFO entry
    (spelling, underscores and capital letters may differ) and rebuild
    DISEASE_INFO / HEALTHY_CLASSES with the exact class names of the model."""
    import difflib

    def norm(text):
        return re.sub(r"[^a-z0-9]", "", text.lower())

    def crop_of(text):
        found = re.findall(r"[A-Za-z]+", text)
        return found[0].lower() if found else ""

    def readable(text):
        parts = [p for p in re.split(r"[_\s]+", text.strip()) if p]
        if not parts:
            return text
        crop = parts[0].title()
        rest = " ".join(parts[1:]).title()
        return f"{crop} - {rest}" if rest else crop

    base = dict(DISEASE_INFO)
    aligned, healthy = {}, set()
    for class_name in class_names:
        is_healthy = "healthy" in class_name.lower()
        entry = base.get(class_name)
        if entry is None:
            same_crop = [k for k in base if crop_of(k) == crop_of(class_name)]
            best, best_score = None, 0.0
            for key in same_crop:
                score = difflib.SequenceMatcher(None, norm(key), norm(class_name)).ratio()
                if score > best_score:
                    best, best_score = key, score
            if best is not None and best_score >= 0.72:
                entry = dict(base[best])
                if norm(best) != norm(class_name):
                    entry["name"] = readable(class_name)  # keep the model's own disease name
        if entry is None:
            title = readable(class_name)
            entry = {
                "name": title,
                "description": (
                    "No disease pattern was detected in the uploaded leaf image."
                    if is_healthy else
                    f"The model detected a pattern similar to {title}. Detailed notes for this class are not available in the app."
                ),
                "symptoms": (
                    "No clear disease symptoms detected by the model."
                    if is_healthy else
                    "Please compare the leaf with the plant and consult an agricultural expert."
                ),
                "treatment": (
                    "No disease treatment is indicated from this prediction."
                    if is_healthy else
                    "Remove badly affected leaves and ask a local agriculture expert about suitable control methods."
                ),
                "precautions": "Inspect the plants regularly and keep the field or garden clean.",
                "prevention": "Use healthy seed or seedlings, keep good spacing, and remove infected plant debris.",
                "fertilizer": "Use balanced fertilizer according to soil testing; fertilizer does not cure disease."
            }
        aligned[class_name] = entry
        if is_healthy:
            healthy.add(class_name)

    if aligned:
        DISEASE_INFO.clear()
        DISEASE_INFO.update(aligned)
        HEALTHY_CLASSES.clear()
        HEALTHY_CLASSES.update(healthy)


_align_disease_info_with_classes()

# =========================================================
# 5. IMAGE PREPROCESSING AND PREDICTION
# =========================================================
def preprocess_image(image):
    image = image.convert("RGB").resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)

def predict_disease(image):
    if model is None:
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Place plant_disease_model.keras inside the models folder."
        )
    if not class_names:
        raise FileNotFoundError(
            f"Class names file not found or empty: {CLASS_NAMES_PATH}"
        )

    prediction = model.predict(preprocess_image(image), verbose=0)[0]
    if len(prediction) != len(class_names):
        raise ValueError(
            f"Model returned {len(prediction)} scores but class_names.txt contains {len(class_names)} labels."
        )

    sorted_indices = np.argsort(prediction)[::-1]
    top_predictions = [
        (class_names[index], float(prediction[index]) * 100)
        for index in sorted_indices[:3]
    ]
    predicted_class = class_names[int(np.argmax(prediction))]
    confidence = float(np.max(prediction)) * 100
    return predicted_class, confidence, top_predictions

# =========================================================
# 5B. EXPLAINABLE AI (GRAD-CAM HEATMAP)
# =========================================================
def _find_last_4d_layer(container):
    """Find the last layer that outputs a feature map (batch, h, w, channels).
    Also searches inside nested models (for example a MobileNet base model)."""
    for layer in reversed(container.layers):
        if isinstance(layer, tf.keras.Model):
            found = _find_last_4d_layer(layer)
            if found is not None:
                return found
            continue
        try:
            if len(layer.output.shape) == 4:
                return container, layer
        except Exception:
            continue
    return None


def make_gradcam_overlay(image, class_index=None):
    """Return an RGB image with a Grad-CAM heatmap over the leaf, or None if unavailable.
    Grad-CAM shows which regions of the leaf pushed the model towards its decision."""
    try:
        if model is None:
            return None
        found = _find_last_4d_layer(model)
        if found is None:
            return None
        container, target_layer = found
        img_array = tf.convert_to_tensor(preprocess_image(image))

        with tf.GradientTape() as tape:
            if container is model:
                grad_model = tf.keras.Model(model.inputs, [target_layer.output, model.output])
                conv_out, preds = grad_model(img_array)
            else:
                inner = tf.keras.Model(container.inputs, [target_layer.output, container.output])
                x = img_array
                conv_out = None
                for layer in model.layers:
                    if isinstance(layer, tf.keras.layers.InputLayer):
                        continue
                    if layer is container:
                        conv_out, x = inner(x)
                    else:
                        x = layer(x)
                preds = x
            if class_index is None:
                class_index = int(tf.argmax(preds[0]))
            score = preds[:, class_index]

        grads = tape.gradient(score, conv_out)
        if grads is None:
            return None
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.squeeze(conv_out[0] @ pooled_grads[..., tf.newaxis])
        heatmap = tf.maximum(heatmap, 0)
        max_value = float(tf.reduce_max(heatmap))
        if max_value <= 0:
            return None
        heatmap = (heatmap / max_value).numpy()

        rgb = np.array(image.convert("RGB"))
        heatmap = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]))
        colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(rgb, 0.6, colored, 0.4, 0)
    except Exception:
        return None

# =========================================================
# 6. DISEASE SEVERITY ESTIMATION
# =========================================================
def estimate_disease_severity(image):
    try:
        rgb = np.array(image.convert("RGB"))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

        # Approximate leaf mask using green/yellow/red plant pixels.
        green_mask = cv2.inRange(hsv, np.array([20, 25, 20]), np.array([100, 255, 255]))
        red_low = cv2.inRange(hsv, np.array([0, 35, 25]), np.array([18, 255, 255]))
        red_high = cv2.inRange(hsv, np.array([160, 35, 25]), np.array([180, 255, 255]))
        leaf_mask = cv2.bitwise_or(green_mask, cv2.bitwise_or(red_low, red_high))

        kernel = np.ones((5, 5), np.uint8)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 500:
            return None

        clean_leaf_mask = np.zeros_like(leaf_mask)
        cv2.drawContours(clean_leaf_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Brown/dark lesion approximation within the detected leaf.
        lab_l = lab[:, :, 0]
        lab_a = lab[:, :, 1]
        lab_b = lab[:, :, 2]
        brown_mask = (
            (lab_l < 155) &
            (lab_a > 120) &
            (lab_b > 120)
        ).astype(np.uint8) * 255
        dark_mask = cv2.inRange(hsv[:, :, 2], 0, 75)
        affected_mask = cv2.bitwise_or(brown_mask, dark_mask)
        affected_mask = cv2.bitwise_and(affected_mask, clean_leaf_mask)
        affected_mask = cv2.morphologyEx(affected_mask, cv2.MORPH_OPEN, kernel)

        leaf_pixels = cv2.countNonZero(clean_leaf_mask)
        affected_pixels = cv2.countNonZero(affected_mask)
        if leaf_pixels == 0:
            return None

        affected_percentage = (affected_pixels / leaf_pixels) * 100
        if affected_percentage <= 10:
            severity = "Low"
        elif affected_percentage <= 30:
            severity = "Medium"
        else:
            severity = "High"

        affected_contours, _ = cv2.findContours(
            affected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return {
            "severity": severity,
            "affected_percentage": affected_percentage,
            "leaf_mask": clean_leaf_mask,
            "affected_mask": affected_mask,
            "affected_contours": affected_contours
        }
    except Exception:
        return None

# =========================================================
# 7. WEATHER LOOKUP (OPEN-METEO)
# =========================================================
def _get_json_with_retry(url, params, attempts=3, timeout=20):
    """GET a JSON response, retrying on network errors, server errors and bad responses."""
    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "PlantDiseaseDetectionApp/1.0"}
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt < attempts - 1:
                time.sleep(1.0 * (attempt + 1))
    raise last_error


def _city_search_names(city):
    """Build a list of spellings to try, e.g. 'Chennai, India' -> ['Chennai, India', 'Chennai']."""
    cleaned = " ".join(city.strip().split())
    names = [cleaned]
    for separator in [",", "-", "/"]:
        if separator in cleaned:
            first_part = cleaned.split(separator)[0].strip()
            if first_part and first_part not in names:
                names.append(first_part)
    return names


def get_weather(city):
    """Return (weather_dict, None) on success or (None, error_message) on failure."""
    try:
        place = None
        for name in _city_search_names(city):
            geo_data = _get_json_with_retry(
                "https://geocoding-api.open-meteo.com/v1/search",
                {"name": name, "count": 5, "language": "en", "format": "json"}
            )
            results = geo_data.get("results") or []
            if results:
                # Prefer the most populated match when several places share a name.
                place = max(results, key=lambda item: item.get("population") or 0)
                break

        if place is None:
            return None, f"City '{city}' was not found. Try another spelling or a nearby big city."

        forecast_url = "https://api.open-meteo.com/v1/forecast"
        base_params = {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "timezone": "auto"
        }
        try:
            weather = _get_json_with_retry(
                forecast_url,
                {
                    **base_params,
                    "current": "temperature_2m,relative_humidity_2m,precipitation",
                    "daily": (
                        "weather_code,temperature_2m_max,temperature_2m_min,"
                        "precipitation_sum,precipitation_probability_max"
                    ),
                    "hourly": "relative_humidity_2m",
                    "forecast_days": 7
                }
            )
        except Exception:
            # If the 7-day request fails, still show today's weather.
            weather = _get_json_with_retry(
                forecast_url,
                {
                    **base_params,
                    "current": "temperature_2m,relative_humidity_2m,precipitation",
                    "daily": "precipitation_probability_max",
                    "forecast_days": 1
                }
            )
        current = weather.get("current") or {}
        daily = weather.get("daily") or {}
        rain_list = daily.get("precipitation_probability_max") or [None]

        return {
            "city": place.get("name", city),
            "country": place.get("country", ""),
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
            "rain_probability": rain_list[0],
            "forecast": _build_forecast(weather)
        }, None
    except Exception as error:
        return None, f"Weather service did not respond ({type(error).__name__}). Please click Get Weather again."


def fmt_value(value, unit=""):
    """Show 'N/A' instead of 'None' when the weather service has no value."""
    return "N/A" if value is None else f"{value}{unit}"


def _build_forecast(weather):
    """Turn the Open-Meteo response into a simple list with one entry per day."""
    daily = weather.get("daily") or {}
    hourly = weather.get("hourly") or {}
    dates = daily.get("time") or []

    # Average the hourly humidity values for each day.
    humidity_by_day = {}
    for stamp, value in zip(hourly.get("time") or [], hourly.get("relative_humidity_2m") or []):
        if value is not None:
            humidity_by_day.setdefault(str(stamp)[:10], []).append(value)

    def pick(key, index):
        values = daily.get(key) or []
        return values[index] if index < len(values) else None

    forecast = []
    for index, date in enumerate(dates):
        humidity_values = humidity_by_day.get(date)
        forecast.append({
            "date": date,
            "code": pick("weather_code", index),
            "t_max": pick("temperature_2m_max", index),
            "t_min": pick("temperature_2m_min", index),
            "rain_mm": pick("precipitation_sum", index),
            "rain_prob": pick("precipitation_probability_max", index),
            "humidity": round(sum(humidity_values) / len(humidity_values)) if humidity_values else None
        })
    return forecast


def weather_icon(code):
    """Pick an emoji for a WMO weather code."""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "🌡️"
    if code == 0:
        return "☀️"
    if code in (1, 2):
        return "🌤️"
    if code == 3:
        return "☁️"
    if code in (45, 48):
        return "🌫️"
    if 51 <= code <= 57:
        return "🌦️"
    if 61 <= code <= 67 or 80 <= code <= 82:
        return "🌧️"
    if 71 <= code <= 77 or code in (85, 86):
        return "❄️"
    if code >= 95:
        return "⛈️"
    return "🌤️"

# =========================================================
# 8. PDF REPORT GENERATION
# =========================================================
def generate_pdf_report(image, result, info, weather_result=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=38, leftMargin=38,
        topMargin=35, bottomMargin=35
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=26,
        textColor=colors.white,
        alignment=1,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#E8FFF1"),
        alignment=1
    )
    section_style = ParagraphStyle(
        "ColorSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=colors.HexColor("#176B45"),
        spaceBefore=11,
        spaceAfter=5
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#263238"),
        spaceAfter=5
    )
    small_style = ParagraphStyle(
        "SmallReport",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#52616B")
    )

    story = []

    banner = Table(
        [[
            Paragraph("🌱 PLANT HEALTH REPORT", title_style)
        ], [
            Paragraph(
                f"🔬 AI-based leaf analysis  |  {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
                subtitle_style
            )
        ]],
        colWidths=[7.15 * inch]
    )
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#167A50")),
        ("TOPPADDING", (0, 0), (-1, 0), 15),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ("TOPPADDING", (0, 1), (-1, 1), 3),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 14),
        ("ROUNDEDCORNERS", [10, 10, 10, 10])
    ]))
    story.extend([banner, Spacer(1, 14)])

    image_buffer = BytesIO()
    image.convert("RGB").save(image_buffer, format="JPEG", quality=90)
    image_buffer.seek(0)

    image_cell = RLImage(image_buffer, width=2.55 * inch, height=1.9 * inch)
    summary_rows = [
        [image_cell, Paragraph(
            f"<b>🎯 Prediction</b><br/>{result['name']}<br/><br/>"
            f"<b>📈 Confidence</b><br/>{result['confidence']:.2f}%<br/><br/>"
            f"<b>🌿 Healthy score</b>  {result['healthy_score']:.2f}%<br/>"
            f"<b>🦠 Disease score</b>  {result['disease_score']:.2f}%",
            body_style
        )]
    ]
    summary = Table(summary_rows, colWidths=[2.8 * inch, 4.35 * inch])
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F8F3")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#B7DEC7")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10)
    ]))
    story.extend([summary, Spacer(1, 10)])

    if result.get("severity"):
        story.append(Paragraph("🌡️ Disease severity", section_style))
        story.append(Paragraph(
            f"{result['severity']} — approximately "
            f"{result.get('affected_percentage', 0):.2f}% affected area.",
            body_style
        ))

    story.append(Paragraph("🏆 Top 3 predictions", section_style))
    top_data = [["Predicted class", "Confidence"]]
    top_data.extend([[name, f"{score:.2f}%"] for name, score in result["top_predictions"]])
    top_table = Table(top_data, colWidths=[5.45 * inch, 1.7 * inch])
    top_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDF2E5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#145A38")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#B8D8C4")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white, colors.HexColor("#F6FBF7")
        ]),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("PADDING", (0, 0), (-1, -1), 7)
    ]))
    story.extend([top_table, Spacer(1, 8)])

    report_sections = [
        ("🦠 About this disease", "description"),
        ("🔍 Symptoms", "symptoms"),
        ("💊 Treatment", "treatment"),
        ("⚠️ Precautions", "precautions"),
        ("🌱 Prevention", "prevention"),
        ("🌾 Fertilizer guidance", "fertilizer")
    ]
    for heading, key in report_sections:
        story.append(Paragraph(heading, section_style))
        story.append(Paragraph(info.get(key, "Not available."), body_style))

    if weather_result:
        story.append(Paragraph("🌦️ Weather information", section_style))
        weather_text = (
            f"<b>📍 Location:</b> {weather_result['city']}, {weather_result['country']}<br/>"
            f"<b>🌡️ Temperature:</b> {weather_result['temperature']} °C &nbsp;&nbsp; "
            f"<b>💧 Humidity:</b> {weather_result['humidity']}%<br/>"
            f"<b>🌧️ Precipitation:</b> {weather_result['precipitation']} mm &nbsp;&nbsp; "
            f"<b>☔ Rain probability:</b> {weather_result['rain_probability']}%"
        )
        story.append(Paragraph(weather_text, body_style))

    footer_box = Table(
        [[Paragraph(
            "<b>⚠️ Note</b><br/>This report is an AI-assisted preliminary screening. "
            "Image quality and similar-looking conditions can affect the prediction. "
            "For confirmation and crop-specific treatment, consult a qualified local "
            "agricultural expert. Follow expert guidance before applying fertilizers "
            "or pesticides.",
            small_style
        )]],
        colWidths=[7.15 * inch]
    )
    footer_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF6DF")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#E7C978")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9)
    ]))
    story.extend([Spacer(1, 14), footer_box])

    doc.build(story)
    buffer.seek(0)
    return buffer

# =========================================================
# 8B. VISUAL HELPER FUNCTIONS (DISPLAY ONLY)
# =========================================================
def get_cause_chip(info, predicted_class):
    """Return a small (emoji + label, colour) chip describing the disease type."""
    text = (info.get("description", "") + " " + predicted_class).lower()
    if "bacterial" in text:
        return "🧫 Bacterial", "#8e24aa"
    if "fungal" in text:
        return "🍄 Fungal", "#ef6c00"
    if "viral" in text or "virus" in text:
        return "🧬 Viral", "#d81b60"
    if "pest" in text or "mite" in text:
        return "🐛 Pest", "#6d4c41"
    return "🦠 Plant disease", "#5e35b1"

def info_card_html(emoji, title, text, color):
    return (
        f'<div class="pd-card" style="border-top:6px solid {color};">'
        f'<div class="pd-card-emoji">{emoji}</div>'
        f'<div class="pd-card-title">{html.escape(title)}</div>'
        f'<div class="pd-card-text">{html.escape(str(text))}</div>'
        '</div>'
    )

def stat_card_html(emoji, label, value, color):
    return (
        f'<div class="pd-stat" style="background:{color};">'
        f'<div class="emoji">{emoji}</div>'
        f'<div class="value">{html.escape(str(value))}</div>'
        f'<div class="label">{html.escape(label)}</div>'
        '</div>'
    )

def care_card_html(emoji, number, text):
    return (
        '<div class="pd-care">'
        f'<div class="head">{emoji} Recommendation {number}</div>'
        f'<div class="body">{html.escape(text)}</div>'
        '</div>'
    )

# =========================================================
# 8C. FORECAST CARDS AND DISEASE RISK ALERT (RULE-BASED)
# =========================================================
def forecast_cards_html(forecast):
    cards = ""
    for index, day in enumerate(forecast):
        try:
            day_date = datetime.strptime(str(day.get("date")), "%Y-%m-%d")
            day_name = "Today" if index == 0 else day_date.strftime("%a")
            day_label = day_date.strftime("%d %b")
        except ValueError:
            day_name, day_label = f"Day {index + 1}", ""
        t_max = "N/A" if day.get("t_max") is None else f"{day['t_max']:.0f}°"
        t_min = "N/A" if day.get("t_min") is None else f"{day['t_min']:.0f}°"
        today_class = " today" if index == 0 else ""
        cards += (
            f'<div class="pd-day{today_class}">'
            f'<div class="dname">{day_name}</div>'
            f'<div class="dline">{day_label}</div>'
            f'<div class="dicon">{weather_icon(day.get("code"))}</div>'
            f'<div class="dtemp">{t_max} <span>/ {t_min}</span></div>'
            f'<div class="dline">☔ {fmt_value(day.get("rain_prob"), "%")}</div>'
            f'<div class="dline">💧 {fmt_value(day.get("humidity"), "%")}</div>'
            '</div>'
        )
    return f'<div class="pd-forecast">{cards}</div>'


def classify_disease_kind(predicted_class, info):
    """Group a disease as viral, pest, bacterial or fungal (fungal is also the default)."""
    text = (
        f"{predicted_class} {info.get('name', '')} {info.get('description', '')}"
    ).lower().replace("_", " ")
    if any(word in text for word in ["virus", "viral", "mosaic", "curl"]):
        return "viral"
    if any(word in text for word in ["mite", "pest", "insect", "whitefly", "caterpillar", "hopper", "aphid"]):
        return "pest"
    if "bacterial" in text:
        return "bacterial"
    return "fungal"


def assess_disease_risk(predicted_class, info, forecast):
    """Estimate Low / Moderate / High disease risk from the 7-day forecast."""
    days = [day for day in forecast if day]
    if not days:
        return None

    temps = [
        (day["t_max"] + day["t_min"]) / 2
        for day in days
        if day.get("t_max") is not None and day.get("t_min") is not None
    ]
    humidities = [day["humidity"] for day in days if day.get("humidity") is not None]
    avg_temp = sum(temps) / len(temps) if temps else None
    avg_humidity = sum(humidities) / len(humidities) if humidities else None
    rainy_days = sum(
        1 for day in days
        if (day.get("rain_prob") or 0) >= 60 or (day.get("rain_mm") or 0) >= 5
    )

    kind = classify_disease_kind(predicted_class, info)
    name_text = f"{predicted_class} {info.get('name', '')}".lower().replace("_", " ")
    points = 0
    max_points = 0
    reasons = []

    if kind in ("fungal", "bacterial"):
        # These diseases spread faster in humid, rainy and warm weather.
        max_points = 5
        if avg_humidity is not None:
            if avg_humidity >= 80:
                points += 2
                reasons.append(("💧", f"Very humid week (average {avg_humidity:.0f}%)"))
            elif avg_humidity >= 65:
                points += 1
                reasons.append(("💧", f"Humid week (average {avg_humidity:.0f}%)"))
            else:
                reasons.append(("🌵", f"Fairly dry air (average {avg_humidity:.0f}%)"))
        if rainy_days >= 4:
            points += 2
            reasons.append(("🌧️", f"{rainy_days} rainy days expected"))
        elif rainy_days >= 2:
            points += 1
            reasons.append(("🌦️", f"{rainy_days} rainy days expected"))
        else:
            reasons.append(("☀️", f"Only {rainy_days} rainy day(s) expected"))
        if avg_temp is not None:
            low, high = (10, 24) if "late blight" in name_text else (20, 32)
            if low <= avg_temp <= high:
                points += 1
                reasons.append(("🌡️", f"Temperature (average {avg_temp:.0f}°C) suits this disease"))
            else:
                reasons.append(("🌡️", f"Temperature (average {avg_temp:.0f}°C) is less suitable for this disease"))
    elif kind == "pest":
        # Mites and many insect pests increase in hot, dry weather.
        max_points = 3
        if avg_humidity is not None:
            if avg_humidity <= 60:
                points += 1
                reasons.append(("🌵", f"Dry air (average {avg_humidity:.0f}%) helps pests increase"))
            else:
                reasons.append(("💧", f"Humid air (average {avg_humidity:.0f}%)"))
        if avg_temp is not None:
            if avg_temp >= 28:
                points += 1
                reasons.append(("🌡️", f"Hot weather (average {avg_temp:.0f}°C)"))
            else:
                reasons.append(("🌡️", f"Average temperature {avg_temp:.0f}°C"))
        if rainy_days <= 1:
            points += 1
            reasons.append(("☀️", f"Only {rainy_days} rainy day(s) expected"))
        else:
            reasons.append(("🌧️", f"{rainy_days} rainy days expected"))
    else:
        # Viral diseases are mostly carried by whiteflies, which like warm, dry weather.
        max_points = 3
        if avg_temp is not None:
            if avg_temp >= 25:
                points += 1
                reasons.append(("🌡️", f"Warm weather (average {avg_temp:.0f}°C) helps whiteflies"))
            else:
                reasons.append(("🌡️", f"Average temperature {avg_temp:.0f}°C"))
        if rainy_days <= 2:
            points += 1
            reasons.append(("☀️", f"Only {rainy_days} rainy day(s) expected"))
        else:
            reasons.append(("🌧️", f"{rainy_days} rainy days expected"))
        if avg_humidity is not None:
            if avg_humidity < 70:
                points += 1
                reasons.append(("🌵", f"Not very humid (average {avg_humidity:.0f}%)"))
            else:
                reasons.append(("💧", f"Humid week (average {avg_humidity:.0f}%)"))

    ratio = points / max_points if max_points else 0.0
    if ratio < 0.34:
        level = "Low"
    elif ratio < 0.67:
        level = "Moderate"
    else:
        level = "High"

    advice = {
        "Low": "✅ The weather this week is not very favourable for this problem. Keep checking your leaves regularly.",
        "Moderate": "⚠️ Check your plants every 2 to 3 days and remove affected leaves early.",
        "High": (
            "🚨 The weather favours fast spread. Check your plants every day, remove affected "
            "leaves, and ask a local agriculture expert about suitable protection."
        )
    }[level]
    extra = {
        "fungal": " Avoid watering in the evening and keep the leaves dry.",
        "bacterial": " Avoid watering in the evening and keep the leaves dry.",
        "pest": " Check the underside of the leaves for tiny pests.",
        "viral": " Watch for whiteflies and remove infected plants early."
    }[kind]
    if level != "Low":
        advice += extra

    return {
        "level": level,
        "ratio": ratio,
        "kind": kind,
        "reasons": reasons,
        "advice": advice
    }


RISK_STYLE = {
    "Low": ("🟢", "#43a047"),
    "Moderate": ("🟠", "#fb8c00"),
    "High": ("🔴", "#e53935")
}


def risk_banner_html(risk, subject):
    emoji, color = RISK_STYLE[risk["level"]]
    pointer = min(max(risk["ratio"] * 100, 6), 94)
    return (
        f'<div class="pd-risk" style="background:{color};">'
        f'<div class="rtitle">{emoji} {risk["level"]} Risk</div>'
        f'<div class="rsub">{html.escape(subject)}</div>'
        '</div>'
        '<div class="pd-gauge pd-gauge-risk">'
        f'<div class="pd-gauge-pointer" style="left:{pointer:.1f}%;"></div>'
        '</div>'
        '<div class="pd-gauge-labels">'
        '<span>🟢 Low</span><span>🟠 Moderate</span><span>🔴 High</span>'
        '</div>'
    )


def risk_reasons_html(reasons):
    chips = "".join(
        f'<span class="pd-reason">{emoji} {html.escape(text)}</span>'
        for emoji, text in reasons
    )
    return f'<div class="pd-reasons">{chips}</div>'

# =========================================================
# 8D. RULE-BASED PLANT CARE CHATBOT (NO API KEY NEEDED)
# =========================================================
CHAT_GREETING = (
    "👋 Hi! I am your plant-care assistant.\n\n"
    "🌿 Ask me about any disease this app knows: **what it is, symptoms, treatment, "
    "prevention or fertilizer**.\n\n"
    "💬 Example: *tomato early blight treatment*"
)

# Words people may type for a crop, mapped to the crop name used in DISEASE_INFO.
CHAT_SYNONYMS = {
    "thakkali": "tomato",
    "தக்காளி": "tomato",
    "maanga": "mango",
    "mangai": "mango",
    "மாம்பழம்": "mango",
    "மாங்காய்": "mango",
    "mamarum": "mango",
    "arisi": "rice",
    "nel": "rice",
    "paddy": "rice",
    "நெல்": "rice",
    "அரிசி": "rice"
}

# Keywords (English, Tanglish and Tamil) that show what the person wants to know.
CHAT_INTENTS = [
    ("description", ["what is", "what's", "cause", "why", "spread"]),
    ("symptoms", ["symptom", "sign", "look like", "looks like", "identify", "recognize",
                  "அறிகுறி", "arikuri", "lakshan"]),
    ("treatment", ["treat", "cure", "medicine", "spray", "control", "remedy", "fungicide",
                   "pesticide", "manage", "மருந்து", "சிகிச்சை", "marundhu", "sigichai"]),
    ("prevention", ["prevent", "avoid", "protect", "precaution", "தடுப்பு", "thadukka", "thadupu"]),
    ("fertilizer", ["fertilizer", "fertiliser", "manure", "nutrient", "nutrition", "urea",
                    "npk", "உரம்", "uram"])
]

CHAT_FIELD_LABELS = {
    "description": ("🦠", "About this disease"),
    "symptoms": ("🔍", "Symptoms"),
    "treatment": ("💊", "Treatment"),
    "precautions": ("⚠️", "Precautions"),
    "prevention": ("🌱", "Prevention"),
    "fertilizer": ("🌾", "Fertilizer guidance")
}

CHAT_INTENT_FIELDS = {
    "description": ["description"],
    "symptoms": ["symptoms"],
    "treatment": ["treatment"],
    "prevention": ["precautions", "prevention"],
    "fertilizer": ["fertilizer"],
    "overview": ["description", "symptoms", "treatment", "precautions", "prevention", "fertilizer"]
}

# Words ignored when matching a question to a disease name.
CHAT_IGNORED_WORDS = {"leaf", "leaves", "and", "the", "of"}


def _chat_tokens(text):
    return set(re.findall(r"[a-z0-9]+", text))


def _match_diseases(tokens):
    """Return (best_score, matching_keys): how many name words appear in the question."""
    best_score = 0
    matches = []
    for key, value in DISEASE_INFO.items():
        name_tokens = {
            word for word in re.findall(r"[a-z0-9]+", value["name"].lower())
            if word not in CHAT_IGNORED_WORDS
        }
        score = len(name_tokens & tokens)
        if score > best_score:
            best_score, matches = score, [key]
        elif score == best_score and score > 0:
            matches.append(key)
    return best_score, matches


def _supported_list_text():
    groups = {}
    for value in DISEASE_INFO.values():
        crop, _, condition = value["name"].partition(" - ")
        groups.setdefault(crop, []).append(condition)
    lines = ["🌱 **I can answer about these crops and conditions:**", ""]
    for crop, conditions in groups.items():
        lines.append(f"- **{crop}**: " + ", ".join(conditions))
    return "\n".join(lines)


def _build_disease_answer(keys, intents):
    wanted_fields = []
    for intent in (intents or ["overview"]):
        for field in CHAT_INTENT_FIELDS[intent]:
            if field not in wanted_fields:
                wanted_fields.append(field)

    parts = []
    for key in keys:
        info = DISEASE_INFO[key]
        lines = [f"**🌿 {info['name']}**"]
        for field in wanted_fields:
            emoji, label = CHAT_FIELD_LABELS[field]
            lines.append(f"{emoji} **{label}:** {info.get(field, 'Not available.')}")
        parts.append("\n\n".join(lines))

    answer = "\n\n---\n\n".join(parts)
    if any(field in wanted_fields for field in ("treatment", "fertilizer")):
        answer += "\n\n⚠️ Please confirm with a local agriculture expert before using any pesticide or fertilizer."
    return answer


def chatbot_reply(question):
    """Answer a question using DISEASE_INFO only (no internet, no API key)."""
    text = question.lower().strip()
    for word, replacement in CHAT_SYNONYMS.items():
        text = text.replace(word, replacement)
    tokens = _chat_tokens(text)

    result = st.session_state.get("analysis_result")
    current_key = None
    if (
        result
        and result.get("confidence", 0) >= UNKNOWN_CONFIDENCE
        and result.get("class") in DISEASE_INFO
    ):
        current_key = result["class"]

    # Greetings and thanks (only for short messages).
    if len(tokens) <= 3 and (tokens & {"hi", "hello", "hey", "hai", "vanakkam"} or "வணக்கம்" in text):
        return CHAT_GREETING
    if len(tokens) <= 4 and (tokens & {"thanks", "thank", "nandri"} or "நன்றி" in text):
        return "😊 You are welcome! Ask me anything else about your plants."

    # What can the assistant answer about?
    if any(phrase in text for phrase in [
        "which crops", "what crops", "which plants", "what plants", "what diseases",
        "which diseases", "supported", "list of", "what can you"
    ]):
        return _supported_list_text()

    # Weather risk from the Weather Information section.
    if any(word in text for word in ["risk", "forecast", "weather", "alert"]):
        risk = st.session_state.get("last_risk")
        if risk:
            emoji, _ = RISK_STYLE[risk["level"]]
            reasons = "\n".join(f"- {r_emoji} {r_text}" for r_emoji, r_text in risk["reasons"])
            return (
                f"{emoji} **{risk['level']} disease risk this week**\n\n"
                f"{reasons}\n\n{risk['advice']}"
            )
        return (
            "🌦️ To see the 7-day disease risk, detect a leaf first, then enter your city "
            "in the **Weather Information** section and click **Get Weather**."
        )

    intents = [name for name, words in CHAT_INTENTS if any(word in text for word in words)]
    if len(intents) > 1 and "description" in intents:
        intents.remove("description")

    score, matches = _match_diseases(tokens)
    if score >= 2:
        keys = matches
        if len(keys) > 1 and current_key in keys:
            keys = [current_key]
    elif current_key and (score == 0 or current_key in matches):
        keys = [current_key]
    elif score >= 1:
        names = [DISEASE_INFO[key]["name"] for key in matches][:8]
        return (
            "🤔 Which one do you mean?\n\n"
            + "\n".join(f"- {name}" for name in names)
            + "\n\n💬 Please type the crop and the disease name."
        )
    else:
        keys = []

    if not keys:
        if intents:
            return (
                "🌿 Which disease? Please type the crop and the disease name "
                "(example: *tomato early blight*), or detect a leaf first and then ask again."
            )
        return (
            "🤖 I can answer questions about plant diseases: **what it is, symptoms, treatment, "
            "prevention and fertilizer**.\n\n"
            "💬 Try: *tomato late blight symptoms*, or ask *which crops do you support?*"
        )

    if len(keys) > 1 and (not intents or len(keys) > 3):
        names = [DISEASE_INFO[key]["name"] for key in keys][:8]
        return (
            "🤔 Which one do you mean?\n\n"
            + "\n".join(f"- {name}" for name in names)
            + "\n\n💬 Please type the full disease name."
        )

    return _build_disease_answer(keys, intents)


def _queue_chat_question(text):
    st.session_state.pending_question = text


def _clear_chat():
    st.session_state.chat_messages = [{"role": "assistant", "content": CHAT_GREETING}]


def render_chatbot():
    st.header("🤖 Ask the Plant Care Assistant")
    st.caption(
        "💬 Ask about a disease: symptoms, treatment, prevention or fertilizer. "
        "Works without internet or any API key."
    )
    if "chat_messages" not in st.session_state:
        _clear_chat()

    with st.container(border=True):
        quick_questions = [
            ("🦠 About", "What is this disease?"),
            ("🔍 Symptoms", "What are the symptoms?"),
            ("💊 Treatment", "How to treat it?"),
            ("🛡️ Prevention", "How to prevent it?"),
            ("🌾 Fertilizer", "Which fertilizer should I use?")
        ]
        button_cols = st.columns(len(quick_questions) + 1)
        for index, (col, (label, question)) in enumerate(zip(button_cols, quick_questions)):
            col.button(label, key=f"quick_question_{index}", on_click=_queue_chat_question, args=(question,))
        button_cols[-1].button("🗑️ Clear chat", key="clear_chat", on_click=_clear_chat)

        history_box = st.container()
        typed_question = st.chat_input("💬 Ask about a plant disease...")

        question = st.session_state.pop("pending_question", None) or typed_question
        if question:
            st.session_state.chat_messages.append({"role": "user", "content": question})
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": chatbot_reply(question)}
            )

        with history_box:
            for message in st.session_state.chat_messages:
                avatar = "🧑‍🌾" if message["role"] == "user" else "🌿"
                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])

# =========================================================
# 9. SESSION STATE
# =========================================================
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "weather_result" not in st.session_state:
    st.session_state.weather_result = None
if "uploaded_image_id" not in st.session_state:
    st.session_state.uploaded_image_id = None
if "last_risk" not in st.session_state:
    st.session_state.last_risk = None

# =========================================================
# 10. MAIN WEBSITE UI
# =========================================================
st.markdown(
    '<div class="pd-hero">'
    '<h1>🌿 Plant Disease Detection</h1>'
    '<p>📸 Upload a clear plant-leaf image to identify a possible disease, '
    '🔍 review its symptoms, and 🌾 get crop-care suggestions.</p>'
    '</div>',
    unsafe_allow_html=True
)

if model is None:
    st.error(
        "❌ Model file is missing. Keep `plant_disease_model.keras` inside the "
        "`models` folder next to this Python file."
    )
if not class_names:
    st.error(
        "❌ Class names are missing. Add `class_names.txt` inside the `models` folder "
        "with one class name per line, in the same order used during model training."
    )

st.markdown("📸 **Take a bright, clear photo of ONE leaf in good daylight** 🌞")

input_method = st.radio(
    "📥 How do you want to add the leaf photo?",
    ["📤 Upload from gallery", "📷 Take photo with camera"],
    horizontal=True
)
if input_method.startswith("📤"):
    uploaded_file = st.file_uploader(
        "📤 Upload a leaf image",
        type=["jpg", "jpeg", "png"]
    )
else:
    uploaded_file = st.camera_input("📷 Take a clear photo of the leaf")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption="🍃 Uploaded leaf image", width="stretch")

    if st.session_state.uploaded_image_id != image_hash:
        st.session_state.analysis_result = None
        st.session_state.weather_result = None
        st.session_state.last_risk = None
        st.session_state.uploaded_image_id = image_hash

    st.info("👇 Please click the button below to detect the plant disease.")
    if st.button("🔍 Detect Plant Disease", type="primary"):
        try:
            with st.spinner("🔬 Analyzing leaf image..."):
                predicted_class, confidence, top_predictions = predict_disease(image)

                info = DISEASE_INFO.get(predicted_class, {
                    "name": predicted_class.replace("_", " "),
                    "description": "Information for this class is not available.",
                    "symptoms": "Please consult an agricultural expert.",
                    "treatment": "Seek local agricultural advice.",
                    "precautions": "Monitor the plant carefully.",
                    "prevention": "Use good crop hygiene and healthy planting material.",
                    "fertilizer": "Use balanced fertilizer based on soil testing."
                })

                # Calculate healthy probability directly from the model output.
                raw_scores = model.predict(preprocess_image(image), verbose=0)[0]
                healthy_score = sum(
                    float(raw_scores[index]) * 100
                    for index, class_name in enumerate(class_names)
                    if class_name in HEALTHY_CLASSES
                )
                disease_score = max(0.0, 100.0 - healthy_score)

                severity_result = None
                if predicted_class not in HEALTHY_CLASSES:
                    severity_result = estimate_disease_severity(image)

                # Explainable AI: heatmap of the regions behind the prediction.
                gradcam_image = make_gradcam_overlay(image, class_names.index(predicted_class))

                st.session_state.analysis_result = {
                    "gradcam_image": gradcam_image,
                    "class": predicted_class,
                    "name": info["name"],
                    "confidence": confidence,
                    "top_predictions": top_predictions,
                    "healthy_score": healthy_score,
                    "disease_score": disease_score,
                    "severity_result": severity_result,
                    "info": info
                }
        except Exception as error:
            st.error(f"❌ Unable to analyze this image: {error}")

    result = st.session_state.analysis_result

    if result is not None:
        info = result["info"]
        predicted_class = result["class"]
        severity_result = result["severity_result"]
        is_healthy = predicted_class in HEALTHY_CLASSES

        # =================================================
        # UNKNOWN LEAF GUARD: no disease name when the AI is unsure
        # =================================================
        if result["confidence"] < UNKNOWN_CONFIDENCE:
            st.divider()
            st.header("🤔 Leaf Not Recognized")
            st.warning(
                "⚠️ This leaf does not look like any plant this AI has learned, so no "
                "disease result is shown. Showing a guess could give wrong advice."
            )
            st.markdown(
                "🌱 **This AI knows:** 🍅 Tomato, 🥭 Mango and 🌾 Rice leaves.\n\n"
                "📸 **Please try again with:** one leaf of these plants, taken close, "
                "in bright daylight, with a plain background."
            )
            st.divider()
            render_chatbot()
            st.markdown(NOTES_HTML, unsafe_allow_html=True)
            st.stop()

        # =================================================
        # RESULT DASHBOARD
        # =================================================
        st.divider()
        st.header("📊 Disease Detection Result")

        donut_col, detail_col = st.columns([1, 2])

        healthy_pct = min(max(result["healthy_score"], 0.0), 100.0)
        with donut_col:
            st.markdown(
                f'<div class="pd-donut" style="background: conic-gradient('
                f'#2e9e5b 0% {healthy_pct:.2f}%, #e0524d {healthy_pct:.2f}% 100%);">'
                f'<div class="pd-donut-inner">{healthy_pct:.0f}%<span>🌿 Healthy</span></div>'
                '</div>'
                f'<div class="pd-legend">🌿 Healthy {result["healthy_score"]:.2f}%'
                f' &nbsp;|&nbsp; 🦠 Disease {result["disease_score"]:.2f}%</div>',
                unsafe_allow_html=True
            )

        with detail_col:
            chips = []
            if result["confidence"] >= HIGH_CONFIDENCE:
                chips.append(('🟢 High confidence', '#2e9e5b'))
            elif result["confidence"] >= LOW_CONFIDENCE:
                chips.append(('🟠 Medium confidence', '#fb8c00'))
            else:
                chips.append(('🔴 Low confidence', '#e53935'))
            if is_healthy:
                chips.append(('✅ Healthy leaf', '#2e9e5b'))
            else:
                chips.append(('🦠 Disease detected', '#e53935'))
                chips.append(get_cause_chip(info, predicted_class))
                if severity_result is not None:
                    sev = severity_result["severity"]
                    sev_emoji = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}.get(sev, "🌡️")
                    sev_color = {"Low": "#43a047", "Medium": "#fb8c00", "High": "#e53935"}.get(sev, "#607d8b")
                    chips.append((f"{sev_emoji} {sev} severity", sev_color))
            chips_html = "".join(
                f'<span class="pd-chip" style="background:{color};">{html.escape(label)}</span>'
                for label, color in chips
            )
            st.markdown(
                '<div class="pd-result-card">'
                f'<div class="pd-result-name">🌿 {html.escape(result["name"])}</div>'
                f'<div>{chips_html}</div>'
                '<div class="pd-bar-row">'
                '<div class="pd-bar-label">🎯 Prediction confidence</div>'
                f'<div class="pd-bar-track"><div class="pd-bar-fill" style="width:{min(max(result["confidence"], 0), 100):.1f}%;"></div></div>'
                f'<div class="pd-bar-value">{result["confidence"]:.2f}%</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

        if result["confidence"] < LOW_CONFIDENCE:
            st.warning(
                "⚠️ The AI is not very sure about this result. Please take a clearer, closer "
                "photo of a single leaf in good daylight and try again. This AI only knows "
                "tomato, mango and rice leaves, so a different plant can give a wrong "
                "result. If unsure, ask an agriculture expert."
            )

        st.subheader("🏆 Top 3 Predictions")
        medals = ["🥇", "🥈", "🥉"]
        bars_html = ""
        for medal, (class_name, score) in zip(medals, result["top_predictions"]):
            readable_name = DISEASE_INFO.get(class_name, {}).get(
                "name", class_name.replace("_", " ")
            )
            width = min(max(score, 0.0), 100.0)
            bars_html += (
                '<div class="pd-bar-row">'
                f'<div class="pd-bar-label">{medal} {html.escape(readable_name)}</div>'
                f'<div class="pd-bar-track"><div class="pd-bar-fill" style="width:{width:.1f}%;"></div></div>'
                f'<div class="pd-bar-value">{score:.2f}%</div>'
                '</div>'
            )
        st.markdown(bars_html, unsafe_allow_html=True)

        # =================================================
        # DISEASE INFORMATION (VISUAL CARDS SIDE BY SIDE)
        # =================================================
        st.divider()
        st.header("🩺 Disease Information")
        st.caption("🔎 Details about the detected plant condition:")

        disease_points = [
            ("🦠", "About this disease", "description", "#7e57c2"),
            ("🔍", "Symptoms", "symptoms", "#ef6c00"),
            ("💊", "Treatment", "treatment", "#1e88e5"),
            ("⚠️", "Precautions", "precautions", "#e53935"),
            ("🌱", "Prevention", "prevention", "#2e9e5b"),
            ("🌾", "Fertilizer guidance", "fertilizer", "#f9a825")
        ]

        for row_start in range(0, len(disease_points), 3):
            card_cols = st.columns(3)
            for col, (emoji, heading, key, color) in zip(
                card_cols, disease_points[row_start:row_start + 3]
            ):
                col.markdown(
                    info_card_html(emoji, heading, info.get(key, "Not available."), color),
                    unsafe_allow_html=True
                )

        # Severity is not estimated for healthy leaves.
        severity_for_care = "Low"
        severity_is_estimated = False

        # =================================================
        # SEVERITY ANALYSIS
        # =================================================
        st.divider()
        st.header("🌡️ Disease Severity Analysis")
        if is_healthy:
            st.success("✅ The model predicted a healthy leaf. Disease severity is not applicable.")
        elif severity_result is None:
            st.warning(
                "⚠️ Severity could not be estimated from this image. "
                "General crop-care suggestions will be shown instead."
            )
        else:
            severity_for_care = severity_result["severity"]
            severity_is_estimated = True
            percentage = severity_result["affected_percentage"]
            sev_emoji = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}.get(severity_for_care, "🌡️")
            pointer_pos = min(max(percentage, 0.0), 100.0)

            st.markdown(
                '<div class="pd-severity-card">'
                '<div class="pd-severity-big">'
                f'<div class="num">{sev_emoji} {percentage:.2f}%</div>'
                f'<div class="lbl">{severity_for_care} severity — estimated affected area</div>'
                '</div>'
                '<div class="pd-gauge-wrap">'
                '<div class="pd-gauge">'
                f'<div class="pd-gauge-pointer" style="left:{pointer_pos:.2f}%;"></div>'
                '</div>'
                '<div class="pd-gauge-labels">'
                '<span>🟢 Low (up to 10%)</span>'
                '<span>🟠 Medium (10–30%)</span>'
                '<span>🔴 High (above 30%)</span>'
                '</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

            # Draw lesion contours on a copy of the original image.
            highlighted = np.array(image.convert("RGB")).copy()
            if severity_result["affected_contours"]:
                cv2.drawContours(
                    highlighted,
                    severity_result["affected_contours"],
                    -1,
                    (255, 0, 0),
                    2
                )
            image_col1, image_col2 = st.columns(2)
            with image_col1:
                st.image(image, caption="🍃 Original image", width="stretch")
            with image_col2:
                st.image(
                    highlighted,
                    caption="🔴 Approximate affected areas highlighted",
                    width="stretch"
                )
            st.caption(
                "🔬 The highlighted regions are an approximate computer-vision estimate, "
                "not a laboratory measurement."
            )

        # =================================================
        # 10B. EXPLAINABLE AI (GRAD-CAM)
        # =================================================
        st.divider()
        st.header("🧠 Why Did the AI Decide This?")
        gradcam_image = result.get("gradcam_image")
        if gradcam_image is None:
            st.info("ℹ️ The AI attention map is not available for this model or image.")
        else:
            st.write(
                "🔥 Red and yellow areas show where the AI looked most while making its "
                "decision. Blue areas had little effect."
            )
            xai_col1, xai_col2 = st.columns(2)
            with xai_col1:
                st.image(image, caption="🍃 Original image", width="stretch")
            with xai_col2:
                st.image(
                    gradcam_image,
                    caption="🔥 AI attention map (Grad-CAM)",
                    width="stretch"
                )
            st.caption(
                "🔬 If the red areas are on the spots or damaged parts of the leaf, "
                "the AI is looking at the right place. If they are on the background, "
                "treat the result with care."
            )

        # =================================================
        # 11. WEATHER INFORMATION
        # =================================================
        st.divider()
        st.header("🌦️ Weather Information")
        st.write("⛅ Weather conditions can help provide context for crop-care suggestions.")
        city_input = st.text_input("🏙️ Enter your city", placeholder="Example: Chennai")
        if st.button("🌦️ Get Weather"):
            if city_input.strip():
                with st.spinner("🛰️ Fetching weather information..."):
                    weather_data, weather_error = get_weather(city_input.strip())
                st.session_state.weather_result = weather_data
                if weather_data is None:
                    st.error(f"❌ {weather_error}")
            else:
                st.warning("⚠️ Please enter a city name.")

        weather_result = st.session_state.weather_result
        if weather_result:
            st.success(
                f"📍 Weather for {weather_result['city']}, {weather_result['country']}"
            )
            w1, w2, w3, w4 = st.columns(4)
            w1.markdown(
                stat_card_html("🌡️", "Temperature", fmt_value(weather_result['temperature'], " °C"), "#ef6c00"),
                unsafe_allow_html=True
            )
            w2.markdown(
                stat_card_html("💧", "Humidity", fmt_value(weather_result['humidity'], "%"), "#1e88e5"),
                unsafe_allow_html=True
            )
            w3.markdown(
                stat_card_html("🌧️", "Precipitation", fmt_value(weather_result['precipitation'], " mm"), "#5e35b1"),
                unsafe_allow_html=True
            )
            w4.markdown(
                stat_card_html("☔", "Rain probability", fmt_value(weather_result['rain_probability'], "%"), "#00897b"),
                unsafe_allow_html=True
            )

            # ---------------------------------------------
            # 7-day forecast and disease risk alert
            # ---------------------------------------------
            forecast = weather_result.get("forecast") or []
            if forecast:
                st.subheader("📅 7-Day Weather Forecast")
                st.markdown(forecast_cards_html(forecast), unsafe_allow_html=True)

                st.subheader("🚨 Disease Risk Alert")
                risk = assess_disease_risk(predicted_class, info, forecast)
                if risk is None:
                    st.info("ℹ️ Not enough forecast data to estimate the disease risk.")
                else:
                    if is_healthy:
                        risk_subject = "Chance of plant diseases developing in the next 7 days"
                    else:
                        risk_subject = f"{result['name']}: chance of spreading in the next 7 days"
                    st.markdown(risk_banner_html(risk, risk_subject), unsafe_allow_html=True)
                    st.markdown(risk_reasons_html(risk["reasons"]), unsafe_allow_html=True)
                    if risk["level"] == "Low":
                        st.success(risk["advice"])
                    elif risk["level"] == "Moderate":
                        st.warning(risk["advice"])
                    else:
                        st.error(risk["advice"])
                    st.caption(
                        "🔬 This alert is a simple weather-based estimate, not a guarantee. "
                        "Always check your plants yourself."
                    )
                    st.session_state.last_risk = {
                        "level": risk["level"],
                        "reasons": risk["reasons"],
                        "advice": risk["advice"]
                    }
            else:
                st.info("ℹ️ The 7-day forecast is not available right now. Click **Get Weather** again.")

        # =================================================
        # 12. SMART CROP CARE RECOMMENDATIONS
        # =================================================
        st.divider()
        st.header("🌱 Smart Crop Care Recommendations")

        if is_healthy:
            st.info(
                "💚 Your leaf was predicted as healthy. These are general plant-maintenance "
                "suggestions, not disease-treatment instructions."
            )
        elif not severity_is_estimated:
            st.caption(
                "ℹ️ Severity could not be estimated, so general monitoring advice is used."
            )

        temperature = weather_result.get("temperature") if weather_result else None
        humidity = weather_result.get("humidity") if weather_result else None
        rainfall = weather_result.get("precipitation") if weather_result else None

        care_recommendations = get_crop_care_recommendation(
            disease_name=predicted_class,
            severity=severity_for_care,
            temperature=temperature,
            humidity=humidity,
            rainfall=rainfall
        )

        care_cards = []
        for number, recommendation in enumerate(care_recommendations, start=1):
            lower_text = recommendation.lower()
            if any(word in lower_text for word in ["water", "moisture", "rain", "drainage"]):
                emoji = "💧"
            elif any(word in lower_text for word in ["air", "humidity", "ventilation", "airflow"]):
                emoji = "🌬️"
            elif any(word in lower_text for word in ["remove", "prune", "infected", "leaf"]):
                emoji = "✂️"
            elif any(word in lower_text for word in ["inspect", "monitor", "check"]):
                emoji = "🔍"
            elif any(word in lower_text for word in ["fertilizer", "nutrition", "soil"]):
                emoji = "🌾"
            elif any(word in lower_text for word in ["heat", "temperature", "shade", "cool"]):
                emoji = "☀️"
            elif any(word in lower_text for word in ["pest", "whitefly", "mite"]):
                emoji = "🐛"
            else:
                emoji = "🌱"
            care_cards.append(care_card_html(emoji, number, recommendation))

        care_col1, care_col2 = st.columns(2)
        for index, card in enumerate(care_cards):
            target_col = care_col1 if index % 2 == 0 else care_col2
            target_col.markdown(card, unsafe_allow_html=True)

        st.caption(
            "💡 Recommendations are general guidance. Follow local agricultural advice "
            "before applying pesticides or changing fertilizer schedules."
        )

        # =================================================
        # 13. DOWNLOAD PDF REPORT
        # =================================================
        st.divider()
        st.header("📄 Download Analysis Report")
        report_data = {
            "name": result["name"],
            "confidence": result["confidence"],
            "healthy_score": result["healthy_score"],
            "disease_score": result["disease_score"],
            "top_predictions": result["top_predictions"],
            "severity": severity_result["severity"] if severity_result else None,
            "affected_percentage": severity_result["affected_percentage"] if severity_result else 0
        }

        try:
            pdf_file = generate_pdf_report(
                image=image,
                result=report_data,
                info=info,
                weather_result=weather_result
            )
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_file,
                file_name="plant_disease_report.pdf",
                mime="application/pdf"
            )
        except Exception as error:
            st.error(f"❌ Could not generate PDF report: {error}")

# =========================================================
# 13B. AI PLANT CARE ASSISTANT (RULE-BASED CHATBOT)
# =========================================================
st.divider()
render_chatbot()

# =========================================================
# 14. NOTES BOX (LIGHT GREEN, ONE ROW, NO HEADING)
# =========================================================
st.markdown(NOTES_HTML, unsafe_allow_html=True)
