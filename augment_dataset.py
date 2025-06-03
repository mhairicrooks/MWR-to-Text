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