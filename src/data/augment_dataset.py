"""
This script provides functions to generate synthetic clinical descriptions
for use in early stages of model development and architecture initialization.

It includes multiple levels of synthetic description generation:
1. A simple, rule-based mapping of class labels to text.
2. More advanced (to be added) context-aware or model-generated synthetic text.
3. Final integration with actual clinical descriptions.

These synthetic descriptions support experiments with language-based inputs
while preserving privacy and reducing dependence on sensitive clinical data.
"""

import numpy as np
import re


def preprocess_conclusion(text):
    if not isinstance(text, str):
        return ""

    # Remove leading numbering or symbols
    text = re.sub(r'^\s*[\d\s\w.,/()-]+/', '', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    # Filter out short or nonsense text (e.g., less than 3 chars or no vowels)
    if len(text) < 3 or not re.search(r'[aeiouAEIOU]', text):
        return ""

    # Capitalize first letter if needed
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text

def add_synthetic_conclusions_straightforward(df):
    """
    Adds a column of simple synthetic clinical conclusions to a DataFrame based on label values.

    This function uses a fixed mapping from integer class labels in the 'r:Th' column
    to short textual descriptions mimicking clinical observations.

    Parameters:
        df (pandas.DataFrame): Input dataframe with a column 'r:Th' containing class labels (0–5).

    Returns:
        pandas.DataFrame: The original dataframe with an added 'Synthetic_Conclusion' column.
    """
    description_map = {
        0: "No thermal changes detected.",
        1: "Slightly elevated temperature.",
        2: "Moderately elevated temperature (surface).",
        3: "Moderately elevated temperature (surface and depth).",
        4: "Increased temperature (surface and depth), partial asymmetry.",
        5: "Increased temperature (surface and depth), greater and clear asymmetry."
    }
    df['Synthetic_Conclusion'] = df['r:Th'].map(description_map)
    return df

def add_synthetic_conclusion_from_temps(df):
    def generate_description(row):
        rTh = int(row['r:Th'])
        parts = []

        # Region values
        deep_R = np.array([row[f"R{i} int"] for i in range(10)])  # R0–R9
        deep_L = np.array([row[f"L{i} int"] for i in range(10)])
        surf_R = np.array([row[f"R{i} sk"] for i in range(10)])
        surf_L = np.array([row[f"L{i} sk"] for i in range(10)])

        # Averages
        mean_deep_R = np.mean(deep_R[:9])
        mean_deep_L = np.mean(deep_L[:9])
        mean_surf_R = np.mean(surf_R[:9])
        mean_surf_L = np.mean(surf_L[:9])

        deep_diff = abs(mean_deep_R - mean_deep_L)
        surf_diff = abs(mean_surf_R - mean_surf_L)

        deep_surf_R = mean_deep_R - mean_surf_R
        deep_surf_L = mean_deep_L - mean_surf_L

        ax_diff_R = deep_R[9] - mean_deep_R
        ax_diff_L = deep_L[9] - mean_deep_L

        hot_spots_R = [i for i in range(9) if deep_R[i] > mean_deep_R + 1]
        hot_spots_L = [i for i in range(9) if deep_L[i] > mean_deep_L + 1]

        # -------------------------------
        # TIERED LOGIC BASED ON r:Th
        # -------------------------------

        if rTh == 0:
            parts.append("No thermal changes detected.")
            if deep_diff > 1.5 or surf_diff > 1.5:
                parts.append("Some inter-breast asymmetry noted, likely physiological.")
            if deep_surf_R > 2 or deep_surf_L > 2:
                parts.append("Internal temps exceed surface, but within low-risk expectations.")
            if hot_spots_R or hot_spots_L:
                parts.append("Minor localized elevations found, consistent with benign findings.")
            return " ".join(parts)

        elif rTh == 1:
            parts.append("Very mild regional irregularities observed.")
            if deep_diff > 0.8 or surf_diff > 0.8:
                parts.append("Mild asymmetry in temperature distribution.")
            if deep_surf_R > 1.2 or deep_surf_L > 1.2:
                parts.append("Slight internal surface disparity.")
            return " ".join(parts)

        elif rTh == 2:
            parts.append("Low-grade thermal anomalies present.")
            if deep_diff > 0.6 or surf_diff > 0.6:
                parts.append("Asymmetry may suggest early localized change.")
            if deep_surf_R > 1.2 or deep_surf_L > 1.2:
                parts.append("Internal temperatures elevated relative to surface.")
            return " ".join(parts)

        elif rTh == 3:
            parts.append("Moderate thermal deviations detected.")
            if deep_diff > 0.5:
                parts.append("Clear asymmetry between breasts.")
            if hot_spots_R or hot_spots_L:
                hs_msg = []
                if hot_spots_R: hs_msg.append(f"R{', R'.join(map(str, hot_spots_R))}")
                if hot_spots_L: hs_msg.append(f"L{', L'.join(map(str, hot_spots_L))}")
                parts.append("Focal regions of elevated activity: " + "; ".join(hs_msg) + ".")
            return " ".join(parts)

        elif rTh >= 4:
            parts.append("Significant thermal irregularities observed.")
            if deep_diff > 0.5:
                parts.append("Marked asymmetry between left and right breast.")
            if deep_surf_R > 1.5 or deep_surf_L > 1.5:
                parts.append("Deep tissue activity likely.")
            if ax_diff_R > 1 or ax_diff_L > 1:
                parts.append("Axillary temperature elevation noted.")
            if hot_spots_R or hot_spots_L:
                hs_msg = []
                if hot_spots_R: hs_msg.append(f"R{', R'.join(map(str, hot_spots_R))}")
                if hot_spots_L: hs_msg.append(f"L{', L'.join(map(str, hot_spots_L))}")
                parts.append("Abnormal focal thermal activity in: " + "; ".join(hs_msg) + ".")
            return " ".join(parts)

        # Fallback
        parts.append("Unrecognized r:Th pattern.")
        return " ".join(parts)

    df['Synthetic_Conclusion'] = df.apply(generate_description, axis=1)
    return df


def add_enhanced_synthetic_conclusions(df):
    """
    Adds a column of detailed synthetic clinical conclusions based ONLY on thermal data.
    These conclusions are clinically accurate and only describe what can be determined
    from temperature measurements, not physical examination findings.

    Parameters:
        df (pandas.DataFrame): Input dataframe with temperature data and 'r:Th' column.

    Returns:
        pandas.DataFrame: The original dataframe with an added 'Enhanced_Synthetic_Conclusion' column.
    """
    
    def generate_realistic_description(row):
        rTh = int(row['r:Th'])
        
        # Extract temperature data
        deep_R = np.array([row[f"R{i} int"] for i in range(10)])  # R0–R9
        deep_L = np.array([row[f"L{i} int"] for i in range(10)])
        surf_R = np.array([row[f"R{i} sk"] for i in range(10)])
        surf_L = np.array([row[f"L{i} sk"] for i in range(10)])

        # Calculate key thermal metrics
        mean_deep_R = np.mean(deep_R[:9])
        mean_deep_L = np.mean(deep_L[:9])
        mean_surf_R = np.mean(surf_R[:9])
        mean_surf_L = np.mean(surf_L[:9])

        deep_diff = abs(mean_deep_R - mean_deep_L)
        surf_diff = abs(mean_surf_R - mean_surf_L)
        
        # Identify hottest regions (only what thermal data shows)
        hot_spots_R = [i for i in range(9) if deep_R[i] > mean_deep_R + 0.5]
        hot_spots_L = [i for i in range(9) if deep_L[i] > mean_deep_L + 0.5]
        
        # Deep vs surface temperature differences
        deep_surf_diff_R = mean_deep_R - mean_surf_R
        deep_surf_diff_L = mean_deep_L - mean_surf_L
        
        parts = []
        
        if rTh == 0:
            # Normal thermal patterns
            if deep_diff < 0.3 and surf_diff < 0.3:
                parts.append("The temperature in both breasts is symmetrically distributed and within normal limits.")
            else:
                parts.append("The temperature distribution shows physiological asymmetry within normal parameters.")
            
            if deep_surf_diff_R > 1.0 or deep_surf_diff_L > 1.0:
                parts.append("The difference between deep and superficial values indicates normal tissue metabolism.")
            
            parts.append("There are no signs of local hyperthermia or abnormal thermal activity.")
            
            # Varied recommendations for normal findings
            recommendation = random.choice([
                "I recommend regular self-examination once a month and systematic thermography in one year.",
                "I advise regular self-examination of the breasts once a month and a systematic examination once a year.",
                "I recommend regular monthly self-examinations of the breasts and a systematic examination once a year.",
                "I advise regular self-examinations immediately after menstruation and a systematic examination once a year."
            ])
            parts.append(recommendation)
            
        elif rTh == 1:
            # Very mild thermal irregularities
            if deep_diff > 0.5:
                parts.append(f"The temperature distribution shows mild asymmetry (difference: {deep_diff:.1f}°C), but remains within acceptable limits.")
            else:
                parts.append("The temperature in both breasts is well distributed with minor variations from the average.")
            
            if surf_diff > 0.4:
                parts.append("Surface temperature patterns show some variation, but no concerning thermal activity.")
            
            # Varied recommendations for mild findings
            recommendation = random.choice([
                "I recommend monthly self-examination and follow-up thermography in 6-12 months.",
                "I recommend self-examination once a month and a systematic check-up in one year or sooner if necessary.",
                "I advise regular self-examinations and a re-measurement in one year.",
                "I recommend regular self-examination and a follow-up breast scan measurement in one year."
            ])
            parts.append(recommendation)
            
        elif rTh == 2:
            # Low-grade thermal anomalies
            parts.append("The temperature distribution is asymmetric, but there are no signs of locally elevated metabolism.")
            
            if deep_diff > 0.6:
                warmer_side = "left" if mean_deep_L > mean_deep_R else "right"
                parts.append(f"The {warmer_side} breast shows slightly elevated deep tissue temperature (asymmetry: {deep_diff:.1f}°C).")
            
            if abs(deep_surf_diff_R - deep_surf_diff_L) > 0.5:
                parts.append("The deep-to-surface temperature gradient varies between breasts, suggesting different thermal patterns.")
            
            # Varied recommendations for low-grade findings
            recommendation = random.choice([
                "I recommend regular self-examinations and clinical evaluation within one year, with follow-up thermography as indicated.",
                "I recommend regular self-examinations, regular clinical examinations by an oncologist, and systematic measurements in one year.",
                "I advise regular self-examinations, regular clinical checks with an oncologist, and systematic measurements in one year.",
                "I recommend regular self-examinations, a clinical examination by an oncologist within one year, and a systematic examination in one year."
            ])
            parts.append(recommendation)
            
        elif rTh == 3:
            # Moderate thermal deviations
            warmer_side = "left" if mean_deep_L > mean_deep_R else "right"
            parts.append(f"The measured temperature shows asymmetric distribution. The {warmer_side} breast demonstrates elevated thermal activity.")
            
            if deep_diff > 0.8:
                parts.append(f"Deep tissue asymmetry is notable (difference: {deep_diff:.1f}°C).")
            
            # Identify specific thermal regions if present
            if hot_spots_R or hot_spots_L:
                region_names = ["upper inner", "upper outer", "lower outer", "lower inner", "central", "upper central", "lower central", "lateral", "medial"]
                if hot_spots_R:
                    regions = [region_names[min(i, len(region_names)-1)] for i in hot_spots_R[:2]]  # Limit to 2 regions
                    parts.append(f"Right breast shows focal thermal elevation in {', '.join(regions)} region(s).")
                if hot_spots_L:
                    regions = [region_names[min(i, len(region_names)-1)] for i in hot_spots_L[:2]]
                    parts.append(f"Left breast shows focal thermal elevation in {', '.join(regions)} region(s).")
            
            # Varied recommendations for moderate findings
            recommendation = random.choice([
                "I recommend clinical examination by a specialist and follow-up thermography in 3-6 months.",
                "I recommend self-examination once a month and a remeasurement in 4-6 months.",
                "I recommend a follow-up in 6-12 months. During this period, we advise monthly self-examinations.",
                "I advise regular self-examination, a clinical examination by an oncologist within one year, and a systematic examination after one year."
            ])
            parts.append(recommendation)
            
        elif rTh == 4:
            # Significant thermal irregularities
            warmer_side = "left" if mean_deep_L > mean_deep_R else "right"
            parts.append(f"The measured temperature demonstrates significant asymmetry and elevation. The {warmer_side} breast shows markedly increased thermal activity.")
            
            # Detailed thermal analysis
            if deep_diff > 1.0:
                region_names = ["upper inner", "upper outer", "lower outer", "lower inner", "central", "upper central", "lower central", "lateral", "medial"]
                max_regions = []
                if warmer_side == "left":
                    max_regions = [region_names[min(i, len(region_names)-1)] for i in range(3) if deep_L[i] > mean_deep_L + 0.8]
                else:
                    max_regions = [region_names[min(i, len(region_names)-1)] for i in range(3) if deep_R[i] > mean_deep_R + 0.8]
                
                if max_regions:
                    temp_values = [f"+{deep_diff + random.uniform(-0.2, 0.4):.1f}°C" for _ in max_regions[:3]]
                    region_details = [f"{reg} ({temp})" for reg, temp in zip(max_regions[:3], temp_values)]
                    parts.append(f"Specific thermal elevation noted in: {', '.join(region_details)}.")
            
            if abs(deep_surf_diff_R - deep_surf_diff_L) > 1.0:
                parts.append("The surface shows contrasting thermal patterns while deep tissue remains highly active.")
            
            # Varied urgent recommendations
            urgent_rec = random.choice([
                "I recommend immediate clinical examination by a specialist physician and comprehensive diagnostic evaluation. Follow-up thermography should be scheduled within 4-6 weeks.",
                "I recommend a clinical examination by a specialist doctor and further diagnostic evaluation as per the doctor's opinion, as well as a follow-up measurement within one month.",
                "We recommend repeating the measurement in 6 months. During this period, we advise monthly self-examinations and preventive ultrasound with a specialist, and, if necessary, also a mammogram.",
                "I advise a clinical examination by a specialized physician and possibly further treatment."
            ])
            parts.append(urgent_rec)
            
        else:  # rTh >= 5
            # High-risk thermal findings
            warmer_side = "left" if mean_deep_L > mean_deep_R else "right"
            parts.append(f"The measured temperature shows marked asymmetry with significantly elevated thermal activity. The {warmer_side} breast demonstrates concerning thermal patterns.")
            
            # Detailed quadrant analysis for high-risk thermal cases
            region_names = ["upper inner quadrant", "upper outer quadrant", "lower outer quadrant", "lower inner quadrant", "central region"]
            elevated_regions = []
            temp_differences = []
            
            for i in range(min(5, len(region_names))):
                if warmer_side == "left" and i < len(deep_L) and deep_L[i] > mean_deep_L + 0.7:
                    elevated_regions.append(region_names[i])
                    temp_differences.append(f"+{deep_L[i] - mean_deep_L:.1f}°C")
                elif warmer_side == "right" and i < len(deep_R) and deep_R[i] > mean_deep_R + 0.7:
                    elevated_regions.append(region_names[i])
                    temp_differences.append(f"+{deep_R[i] - mean_deep_R:.1f}°C")
            
            if elevated_regions:
                region_details = [f"{reg} ({temp})" for reg, temp in zip(elevated_regions[:3], temp_differences[:3])]
                parts.append(f"Multiple regions show significant thermal elevation: {', '.join(region_details)}.")
            
            if mean_surf_R < mean_deep_R - 1.5 or mean_surf_L < mean_deep_L - 1.5:
                parts.append("Surface cooling observed despite deep tissue thermal activity, indicating possible vascular involvement.")
            
            # Varied high-risk recommendations
            high_risk_rec = random.choice([
                "These thermal findings require immediate clinical correlation and comprehensive diagnostic workup including imaging studies. Urgent follow-up thermography recommended within 2-3 weeks, with earlier evaluation if clinical symptoms develop.",
                "I recommend immediate clinical examination by a specialist doctor and comprehensive diagnostic evaluation including ultrasound and mammography if deemed necessary. A follow-up measurement should be scheduled within 2-4 weeks.",
                "We recommend repeating the measurement in 6 months after clinical evaluation. During this period, we advise monthly self-examinations and preventive check-ups with a specialist.",
                "I recommend a clinical examination by a specialist doctor and further diagnostic evaluation, as well as a follow-up measurement within one month."
            ])
            parts.append(high_risk_rec)
        
        # Add general advisory for moderate to high-risk thermal findings
        if rTh >= 3:
            parts.append("In case of any palpable changes, discharge, or persistent symptoms, seek immediate clinical evaluation.")
        
        return " ".join(parts)
    
    df['Enhanced_Synthetic_Conclusion'] = df.apply(generate_realistic_description, axis=1)
    return df
