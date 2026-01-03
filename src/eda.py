import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


class ComplainEDAProcessor:
    """
    Handles EDA and preprocessing for CFPB complaint data.
    """
    TARGET_PRODUCTS = [
    "Credit card",
    "Personal loan",
    "Savings account",
    "Money transfer"
]

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df=None
    # -------------Data Loading-----------------
    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print(f"Data Loaded: with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")
        return self.df
    

    # -------------EDA Functions-----------------
    def dataset_overview(self):
        """Dispaly basic informatio about the dataset """
        self.df.head()
        print("Dataset Overview:")
        print(self.df.info())
        print("\nMissing Values:")
        print(self.df.isnull().sum())
    def plot_product_distribution(self):
        """Plot complaint distribtion by product tyepe"""
        plt.figure(figsize=(10,6))
        self.df["Product"].value_counts().plot(kind= "bar")
        plt.title("Complaint Distribution by product type")
        plt.xlabel("Product Type")
        plt.ylabel("Number of Complaints")
        plt.show()
    def add_narrative_length(self):
        """Add a new column with the length of the complaint narrative"""
        self.df["narrative_length"] = self.df["Consumer complaint narrative"].fillna("").apply(lambda x: len(x.split()))
        print("Added 'narrative_length' column.")
        return self.df

    def plot_narrative_length_distribution(self):
        """plot distribution of narrative lengths"""
        plt.figure(figsize=(10,6))
        sns.histplot(self.df["narrative_length"], bins=30, kde=True)
        plt.title("Distribtion of Complaint Narrative Lengths")
        plt.xlabel("Narrative Length (words)")
        plt.ylabel("Frequency")
        plt.show()
    def narrative_presence_stats(self):
        """ identify conplaints with and without narratives"""
        with_narrative=self.df["Consumer complaint narrative"].notnull().sum()
        without_narrative=self.df["Consumer complaint narrative"].isnull().sum()
        print(f"Complaints with Narrative: {with_narrative}")
        print(f"Complaints without Narrative: {without_narrative}")
    #----------Filter the dataset for target products-------------
    def filter_product(self):
        """filter dataset for Target Products"""
        self.df=self.df[self.df["Product"].isin(self.TARGET_PRODUCTS)]
        print(f"Filtered dataset to {self.df.shape[0]} rows for target products.")
        return self.df
    def remove_empty_narratives(self):
        """Remove rows with empty complaint narratives"""
        self.df.dropna(subset=["Consumer complaint narrative"],inplace=True)
        self.df=self.df[self.df["Consumer complaint narrative"].str.strip()!=""]# remove white space only narratives
        print(f"Removed empty narrative rows. Remaining rows: {self.df.shape[0]}")
        return self.df
    #------Clean the text narratives to improve embedding quality--------
    @staticmethod
    def clean_text(text:str)->str:
        """Clean Complaint Narrative Text"""
        text=text.lower() # Lower case
        text=re.sub(r"i am writing to file  acomplaint","",text) # Remove common phrases
        text=re.sub(r"[^a-zA-Z0-9\s]","",text)# remove spacial characters
        text=re.sub(r"\s+"," ",text).strip() # remove extra spaces
        return text

    def apply_text_cleaning(self):
        """Apply text cleaning to the complaint narratives"""
        self.df["cleaned_narrative"]=self.df["Consumer complaint narrative"].apply(self.clean_text)
    # =-----------------Save processed data-----------------
    def save_processed_data(self, Output_path:str):
        """Save the processed dataframe to a CSV file"""
        self.df.to_csv(Output_path,index=False)
        print(f"Processed data Saved to {Output_path}")