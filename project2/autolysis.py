import matplotlib
matplotlib.use('Agg')  

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def load_data(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        print("Error loading file:", e)
        sys.exit()

def basic_analysis(df):
    summary = df.describe(include='all').to_string()
    missing = df.isnull().sum().to_string()
    return summary, missing

def correlation_plot(df):
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] > 1:
        plt.figure()
        sns.heatmap(numeric_df.corr(), annot=True)
        plt.title("Correlation Heatmap")
        plt.savefig("chart1.png")
        plt.close()

def distribution_plot(df):
    numeric_df = df.select_dtypes(include=np.number)
    if len(numeric_df.columns) > 0:
        col = numeric_df.columns[0]
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig("chart2.png")
        plt.close()

def generate_story(summary, missing, columns):
    story = f"""
# 📊 Automated Data Analysis Report

## 📁 Dataset Overview
The dataset contains the following columns:
{columns}

## 🔍 Summary Statistics
{summary}

## ❗ Missing Values
{missing}

## 📈 Insights
- Dataset shows distribution patterns across variables
- Correlation between features is visualized
- Some columns may have missing values affecting results

## 💡 Implications
- Clean missing data before modeling
- Focus on highly correlated features
- Use insights for decision-making
"""
    return story

def save_readme(story):
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(story)

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py dataset.csv")
        sys.exit()

    file = sys.argv[1]
    df = load_data(file)

    summary, missing = basic_analysis(df)

    correlation_plot(df)
    distribution_plot(df)

    story = generate_story(summary, missing, df.columns)
    save_readme(story)

    print("Analysis complete. README.md and charts generated.")

if __name__ == "__main__":
    main()