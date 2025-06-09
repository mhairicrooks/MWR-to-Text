"""
Data Loading and Selection Module for Thermal Asymmetry Analysis

This module defines two main classes:

1. DataLoader:
   - Responsible for loading, preprocessing, and cleaning breast temperature and clinical data.
   - Includes methods for handling missing values, casting column types, normalizing temperature readings,
     computing cycle stages, and selecting subsets of data columns based on anatomical regions (left/right breast),
     measurement surfaces (skin/depth), and reference or axillary values.
   - Provides utilities to shuffle data and filter by asymmetry.

2. DataLoaderSelector:
   - Provides a high-level interface to customize and retrieve data subsets tailored to specific analysis needs.
   - Supports flexible selection of glands (left/right/both), surfaces (skin/depth/both), and optional inclusion of
     reference values, axillary measurements, and patient age.
   - Returns feature matrices, asymmetry classification labels, and corresponding clinical text descriptions,
     supporting downstream machine learning or statistical modeling workflows.

Usage:
- Instantiate DataLoaderSelector with the dataset path.
- Call `get_data()` with desired parameters to obtain feature data (X), asymmetry labels (y_class),
  and clinical text labels (y_text).

This modular design supports extensible and customizable data preparation for breast temperature asymmetry studies,
including both numerical and text data targets.
"""

import re
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np

from scipy.optimize import curve_fit


class DataLoader():
    
    def __init__(self, data_path: str):
        """
        Initialize the DataLoader object.

        Args:
            data_path (str): The path to the data file.

        Returns:
            None
        """
        if not data_path is None:
            self.data = pd.read_csv(data_path)
            self.data = DataLoader.prepare_data(self.data)
            self.data = DataLoader.drop_nans(self.data)
            self.data = DataLoader.cast_types(self.data)
            
        self.asymmetry = 'r:Th'
        self.l_values = ['L{} int'.format(i) for i in range(10)]  # L0 int to L9 int
        self.r_values = ['R{} int'.format(i) for i in range(10)]  # R0 int to R9 int
        self.l_skin_values = ['L{} sk'.format(i) for i in range(10)]  # L0 sk to L9 sk
        self.r_skin_values = ['R{} sk'.format(i) for i in range(10)]  # R0 sk to R9 sk
        self.l_axillary_value = 'L9 int'
        self.r_axillary_value = 'R9 int'
        self.l_axillary_skin_value = 'L9 sk'
        self.r_axillary_skin_value = 'R9 sk'
        self.ref_values = ['T1 int', 'T2 int']
        self.ref_skin_values = ['T1 sk', 'T2 sk']
        
    
    @staticmethod
    def prepare_data(df: pd.DataFrame, show_menopause: bool = True) -> pd.DataFrame:
        """
        Preprocesses the given DataFrame by filling missing values and transforming the 'Cycle' and 'Day of Cycle' columns.

        Args:
            df (pd.DataFrame): The input DataFrame to be processed.
            show_menopause (bool, optional): Whether to show menopause feature. Defaults to True.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        df['Ambient temperature'] = df['Ambient temperature'].fillna((df['Ambient temperature'].mean()))
        df['Mammary diameter'] = df['Mammary diameter'].fillna((df['Mammary diameter'].mean()))

        pattern = re.compile("^([0-9]+-[0-9]+)+$")

        for index, row in df.iterrows():
            if 'Cycle' in df.columns:
                cycle_val = row['Cycle']
            
                # Check if value is null, empty, or dash
                if pd.isnull(cycle_val) or cycle_val == '' or cycle_val == '-':
                    if show_menopause:
                        df.at[index, 'Cycle'] = -1
                    else:
                        df.at[index, 'Cycle'] = 28
                # Check if it's a string that matches the range pattern (e.g., "25-30")
                elif isinstance(cycle_val, str) and pattern.match(cycle_val):
                    values = cycle_val.split('-')
                    average = int((int(values[0]) + int(values[1])) * 0.5)
                    df.at[index, 'Cycle'] = average
                # If it's already a number, keep it as is
                elif isinstance(cycle_val, (int, float)) and not pd.isnull(cycle_val):
                    df.at[index, 'Cycle'] = int(cycle_val)
                # Handle any other unexpected cases
                else:
                    if show_menopause:
                        df.at[index, 'Cycle'] = -1
                    else:
                        df.at[index, 'Cycle'] = 28

            # Day from the first day
            if 'Day from the first day' in df.columns:
                day_val = row['Day from the first day']
            
                if pd.isnull(day_val) or day_val == '' or day_val == '-':
                    if show_menopause:
                        df.at[index, 'Day from the first day'] = -1
                    else:
                        df.at[index, 'Day from the first day'] = 5
                # If it's already a number, keep it as is
                elif isinstance(day_val, (int, float)) and not pd.isnull(day_val):
                    df.at[index, 'Day from the first day'] = int(day_val)
                # Handle any other unexpected cases
                else:
                    if show_menopause:
                        df.at[index, 'Day from the first day'] = -1
                    else:
                        df.at[index, 'Day from the first day'] = 5
        return df
            
    
    @staticmethod
    def cast_types(df: pd.DataFrame) -> pd.DataFrame:
        """
        Casts the data types of specific columns in the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with updated data types.
        """
        df = df.astype({'r:AgeInYears': int,
                        'r:Th': int,
                        'Mammary diameter': int,
                        'Cycle': int,
                        'Day from the first day': int,
                        # 'Cycle_Stage': int,
                        'Ambient temperature': int})
            
        return df
        
    
    @staticmethod
    def compute_cycle(df: pd.DataFrame, groups: float) -> pd.DataFrame:
        """
        Compute the cycle stage for each row in the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            groups (float): The number of groups to divide the cycle stage into.

        Returns:
            pd.DataFrame: The DataFrame with an additional column 'Cycle_Stage' representing the cycle stage for each row.
        """
        def compute(cycle: int, cycle_day: int) -> float:
            if cycle < 0 or cycle_day < 0:
                return -1 
            
            # For percentage purposes do not let the cycle day exceed the expected cycle length
            cycle_day = min(cycle, cycle_day)
            
            cycle_stage = cycle_day / cycle
            if groups:
                cycle_stage = int(cycle_stage // groups)
                
            return cycle_stage
        
        df['Cycle_Stage'] = df.apply(lambda x: compute(int(x['Cycle']), int(x['Day from the first day'])), axis=1)
        
        return df
    
        
    @staticmethod
    def noramalise(df: pd.DataFrame, label_tag: str, ref_label: str) -> pd.DataFrame:
        """
        Normalize the temperature data in the given DataFrame using a linear transformation
        based on the reference temperature values.

        Args:
            df (pd.DataFrame): The DataFrame containing the temperature data.
            label_tag (str): The tag used to identify the temperature columns in the DataFrame.
            ref_label (str): The label of the reference temperature column.

        Returns:
            pd.DataFrame: The DataFrame with the normalized temperature data.
        """
        def line_function(x, A, B):
            return A * x + B

        def transform(temperature, A, refAvg, ref):
            return temperature + A * (refAvg - ref)
        
        
        ref_mean = df[ref_label].mean(axis=0)
        
        for i in range(10):
            label = f"{label_tag}{i} int" # e.g. R0 int, L0 int
            if label in df.columns:
                A, B = curve_fit(line_function, df[ref_label].values, df[label].values)[0]
                df[label] = np.vectorize(transform)(df[label], A, ref_mean, df[ref_label])
        return df
    
    
    @staticmethod
    def select_asymmetry_cases(df: pd.DataFrame, has_asymmetry: bool) -> pd.DataFrame:
        """
        Selects cases based on whether they have any asymmetry (r:Th > 0) or not.

        Args:
            df (pd.DataFrame): The DataFrame.
            has_asymmetry (bool): True to select cases with asymmetry (r:Th > 0), False for no asymmetry (r:Th == 0).

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        if has_asymmetry:
            return df.loc[df['r:Th'] > 0]
        else:
            return df.loc[df['r:Th'] == 0]
    
    
    @staticmethod
    def drop_nans(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows with missing values from a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to drop rows from.

        Returns:
            pd.DataFrame: The DataFrame with rows containing missing values dropped.
        """
        return df.dropna(axis=0, how='any')
        
    
    def select_columns(self, df: pd.DataFrame, l_values: bool = False,
                       r_values: bool = False, l_skin_values: bool = False,
                       r_skin_values: bool = False, l_axil_value: bool = False,
                       r_axil_value: bool = False, l_axil_skin_value: bool = False,
                       r_axil_skin_value: bool = False, age: bool = False,
                       ref_values: bool = False, ref_skin_values: bool = False,
                       asymmetry_values: bool = False) -> pd.DataFrame:
        """
        Selects specific columns from a DataFrame based on the provided arguments.

        Args:
            df (pd.DataFrame): The input DataFrame.
            l_values (bool, optional): Whether to include the left depth breast temperatures. Defaults to False.
            r_values (bool, optional): Whether to include the right depth breast temperatures. Defaults to False.
            l_skin_values (bool, optional): Whether to include the left skin breast temperatures. Defaults to False.
            r_skin_values (bool, optional): Whether to include the right skin breast temperatures. Defaults to False.
            l_axil_value (bool, optional): Whether to include the left axillary temperature. Defaults to False.
            r_axil_value (bool, optional): Whether to include the right axillary temperature. Defaults to False.
            l_axil_skin_value (bool, optional): Whether to include the left axillary skin temperature. Defaults to False.
            r_axil_skin_value (bool, optional): Whether to include the right axillary skin temperature. Defaults to False.
            age (bool, optional): Whether to include the age. Defaults to False.
            ref_values (bool, optional): Whether to include the reference temperatures. Defaults to False.
            ref_skin_values (bool, optional): Whether to include the reference skin temperatures. Defaults to False.
            asymmetry_values (bool, optional): Whether to include the 'asymmetry' column. Defaults to False.

        Returns:
            pd.DataFrame: A new DataFrame containing only the selected columns.
        """
        column_selection = []
        
        if age:
            column_selection.append('r:AgeInYears')
        if l_values:
            column_selection.extend(self.l_values)
        if r_values:
            column_selection.extend(self.r_values)
        if l_skin_values:
            column_selection.extend(self.l_skin_values)
        if r_skin_values:
            column_selection.extend(self.r_skin_values)
        if l_axil_value:
            column_selection.append(self.l_axillary_value)
        if r_axil_value:
            column_selection.append(self.r_axillary_value)
        if l_axil_skin_value:
            column_selection.append(self.l_axillary_skin_value)
        if r_axil_skin_value:
            column_selection.append(self.r_axillary_skin_value)
        if ref_values:
            column_selection.extend(self.ref_values)
        if ref_skin_values:
            column_selection.extend(self.ref_skin_values)
        if asymmetry_values:
            column_selection.append(self.asymmetry)
        
        return df[column_selection]
    
    @staticmethod
    def shuffle(df: pd.DataFrame, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Shuffle the rows of a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be shuffled.
            seed (int, optional): The random seed for shuffling. Defaults to None.

        Returns:
            pd.DataFrame: The shuffled DataFrame.
        """
        # df = df.shuffle(frac=1).reset_index(drop=True)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        return df
    
    
    
class DataLoaderSelector():
    
    def __init__(self, data_path: str = 'dataset.csv'):
        """
        Initialize the DataLoader class.

        Args:
            data_path (str, optional): The path to the dataset file. Defaults to 'dataset.csv'.
        
        Returns:
            None
        """
        self.data_loader = DataLoader(data_path)
    
    
    @staticmethod
    def _select_data(gland: str = 'both', surface: str = 'both',
                     use_ref_values: bool = True, use_axillary_values: bool = True,
                     use_age: bool = False) -> Dict[str, bool]:
        """
        Returns the appropriate data selection criteria based on the input.

        Args:
            gland (str, optional): The gland to select data from. Valid values are 'both' (default),
                                   'left' or 'l', and 'right' or 'r'.
            surface (str, optional): The surface to select data from. Valid values are 'both' (default),
                                     'skin' or 's', and 'depth' or 'd'.
            use_ref_values (bool, optional): Whether to use reference values. Defaults to True.
            use_axillary_values (bool, optional): Whether to use axillary values. Defaults to True.
            use_age (bool, optional): Whether to use age. Defaults to False.

        Raises:
            ValueError: If invalid values are provided for the 'gland' or 'surface' arguments.

        Returns:
            Dict[str, bool]: A dictionary containing the selected data criteria.
        """
        input_keys = dict(locals())
        
        gland = gland.lower()
        surface = surface.lower()
        
        if surface == 'skin' or surface == 's':
            l_values = False
            r_values = False
            ref_values = False
            l_axil_value = False
            r_axil_value = False
            l_skin_values = True
            r_skin_values = True
            ref_skin_values = True
            l_axil_skin_value = True
            r_axil_skin_value = True
        elif surface == 'depth' or surface == 'd':
            l_values = True
            r_values = True
            ref_values = True
            l_axil_value = True
            r_axil_value = True
            l_skin_values = False
            r_skin_values = False
            ref_skin_values = False
            l_axil_skin_value = False
            r_axil_skin_value = False
        elif surface == 'both' or surface == 'b':
            l_values = True
            r_values = True
            ref_values = True
            l_axil_value = True
            r_axil_value = True
            l_skin_values = True
            r_skin_values = True
            ref_skin_values = True
            l_axil_skin_value = True
            r_axil_skin_value = True
        else:
            raise ValueError(f"{surface} is not a valid input for 'surface' argument") 
        
        
        if gland == 'both' or gland == 'b': # values will be as set by surface condition
            pass
        elif gland == 'left' or gland == 'l':
            r_values = False
            r_skin_values = False
            r_axil_value = False
            r_axil_skin_value = False
        elif gland == 'right' or gland == 'r':
            l_values = False
            l_skin_values = False
            l_axil_value = False
            l_axil_skin_value = False
        else:
            raise ValueError(f"{gland} is not a valid input for 'gland' argument") 
        
        if not use_ref_values:
            ref_values = False
            ref_skin_values = False
        
        if not use_axillary_values:
            l_axil_value = False
            l_axil_skin_value = False
            r_axil_value = False
            r_axil_skin_value = False
        
        if use_age:
            age = True
        else:
            age = False
        
        selections = dict(locals())
        # Remove input variables
        for key in input_keys:
            del selections[key]
        del selections['input_keys']
        
        return selections
    
    
    def get_data(self, gland: str = 'both', surface: str = 'both',
             use_ref_values: bool = True, use_axillary_values: bool = True,
             use_age: bool = False, text_column: str = 'Conclusion (Tr)') -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Get the data based on the specified parameters.

        Args:
            gland (str, optional): The type of gland to include in the data. Defaults to 'both'.
            surface (str, optional): The type of surface to include in the data. Defaults to 'both'.
            use_ref_values (bool, optional): Whether to include reference values in the data. Defaults to True.
            use_axillary_values (bool, optional): Whether to include axillary values in the data. Defaults to True.
            use_age (bool, optional): Whether to include age in the data. Defaults to False.

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.Series]: A tuple containing the features (X),
                                                   asymmetry labels (y_class), and
                                                   clinical text descriptions (y_text).
        """
        selections = DataLoaderSelector._select_data(
            gland=gland,
            surface=surface,
            use_ref_values=use_ref_values,
            use_axillary_values=use_axillary_values,
            use_age=use_age
        )

        # Features
        X = self.data_loader.select_columns(self.data_loader.data, **selections, asymmetry_values=False)

        # Classification target
        # multiclass approach
        y_class = self.data_loader.data['r:Th']
       

        # Text generation target
        if text_column not in self.data_loader.data.columns:
            raise ValueError(f"Column '{text_column}' not found in data.")
        y_text = self.data_loader.data[text_column]

        return X, y_class, y_text