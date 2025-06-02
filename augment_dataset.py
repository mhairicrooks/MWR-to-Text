def add_synthetic_conclusions_straightforward(df):
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