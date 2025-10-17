import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set global plotting style
sns.set_theme(style="whitegrid")

# --- 1. Helper Functions ---

@st.cache_data
def clean_and_split_cuisines(cuisine_string):
    """Splits a string of cuisines into a list of cleaned, title-cased cuisine names."""
    if pd.isna(cuisine_string):
        return []
    return [c.strip().title() for c in cuisine_string.split(',')]

# --- 2. Data Loading and Feature Engineering (Cached for Performance) ---

@st.cache_data
def load_and_preprocess_data(filepath='final_robust_data.csv'):
    """Loads data and performs all necessary feature engineering."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"Error: '{filepath}' not found. Please ensure the file is in the same directory.")
        return None

    # Feature Engineering 1: Cost Range Feature
    cost_quantiles = df['Cost'].quantile([0.33, 0.66]).tolist()
    cost_bins = [df['Cost'].min()] + cost_quantiles + [df['Cost'].max() + 1]
    cost_labels = ['Low Cost (Budget-friendly)', 'Medium Cost (Mid-range)', 'High Cost (Premium)']
    df['Cost_Range'] = pd.cut(df['Cost'], bins=cost_bins, labels=cost_labels, right=False)

    # Feature Engineering 2: Cuisine List Feature (Required for filtering)
    df['Cuisine_List'] = df['Cuisine'].apply(clean_and_split_cuisines)
    
    return df

# --- 3. The Recommendation Model Function ---

def get_recommendations(
    df: pd.DataFrame,
    target_cuisine: str = None,
    target_locality: str = None,
    target_cost_range: str = None,
    sort_by: str = 'Final_Robust_Score',
    top_n: int = 10
) -> pd.DataFrame:
    """
    Generates a list of recommended restaurants based on user-defined criteria.
    """
    recommendations = df.copy()
    
    # 1. Apply Filters
    
    if target_cuisine:
        target_cuisine_title = target_cuisine.strip().title()
        recommendations = recommendations[
            recommendations['Cuisine_List'].apply(lambda x: target_cuisine_title in x)
        ]
        
    if target_locality and target_locality != "Any":
        recommendations = recommendations[
            recommendations['Locality'].str.contains(target_locality, case=False, na=False)
        ]

    if target_cost_range and target_cost_range != "Any":
        # Check if the category exists to prevent error
        if target_cost_range in df['Cost_Range'].cat.categories:
             recommendations = recommendations[
                recommendations['Cost_Range'] == target_cost_range
            ]

    # Handle case with no results after filtering
    if recommendations.empty:
        st.warning("No exact matches found for the specified criteria. Showing top-ranked alternatives.")
        # Fallback to the top 10 overall by Robust Score
        return df.sort_values(by='Final_Robust_Score', ascending=False).head(10)[
            ['Name', 'Locality', 'Cuisine', 'Rating', 'Cost', 'Final_Robust_Score']
        ]

    # 2. Sort and Rank the Results
    ranked_recommendations = recommendations.sort_values(
        by=sort_by,
        ascending=False
    ).head(top_n)

    # 3. Final Output
    return ranked_recommendations[
        ['Name', 'Locality', 'Cuisine', 'Rating', 'Cost', sort_by]
    ].rename(columns={sort_by: f"Ranked by: {sort_by}"})

# --- 4. Streamlit App Structure ---

def main():
    st.title("🍽️ Robust Restaurant Recommendation Engine")
    st.markdown("Use the sidebar to define your preferences and generate personalized recommendations based on our advanced scoring model.")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    if df is None:
        return

    # --- 4.1 Extract unique values for filters ---
    unique_localities = sorted(df['Locality'].unique().tolist())
    all_cuisines_flat = df['Cuisine_List'].explode().dropna().unique()
    unique_cuisines = sorted(all_cuisines_flat.tolist())
    
    cost_options = df['Cost_Range'].cat.categories.tolist()
    sort_options_map = {
        "Overall Quality (Robust Score)": "Final_Robust_Score",
        "Health Focus": "Health_Score_New",
        "Customer Rating": "Rating",
        "Cost (Descending)": "Cost",
    }
    
    # --- 4.2 Sidebar for Recommendation Inputs ---
    
    with st.sidebar:
        st.header("🔍 Recommendation Filters")
        
        # Cuisine Autocomplete/Suggestion Input
        target_cuisine = st.selectbox(
            "Cuisine", 
            options=[""] + unique_cuisines,
            index=0,
            help="Select a specific cuisine for the recommendation."
        )

        # Locality Selectbox
        target_locality = st.selectbox(
            "Locality/Area",
            options=["Any"] + unique_localities,
            index=0
        )

        # Cost Range Selectbox
        target_cost_range = st.selectbox(
            "Cost Range",
            options=["Any"] + cost_options,
            index=0
        )

        st.subheader("Ranking Priority")
        
        # Sorting Criteria
        sort_by_label = st.selectbox(
            "Sort Results By:",
            options=list(sort_options_map.keys()),
            index=0
        )
        sort_by_column = sort_options_map[sort_by_label]

        # Number of Recommendations
        top_n = st.slider("Show Top N Results", min_value=5, max_value=50, value=10)
        
        st.markdown("---")
        st.info("The **Overall Quality (Robust Score)** is the best metric, as it incorporates Trust, Votes, and Rating.")

    # --- 4.3 Main Content Tabs ---
    
    tab1, tab2 = st.tabs(["⭐ Recommendations", "📊 Detailed Analysis & Visuals"])
    
    with tab1:
        st.subheader(f"Top {top_n} Recommended Restaurants")
        
        # Run the Recommendation Model
        recommendations_df = get_recommendations(
            df,
            target_cuisine=target_cuisine,
            target_locality=target_locality,
            target_cost_range=target_cost_range,
            sort_by=sort_by_column,
            top_n=top_n
        )
        
        st.dataframe(recommendations_df, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Ranking Metric Descriptions")
        st.markdown(f"""
        - **Final_Robust_Score**: Our composite quality metric based on trust, votes, and rating. (Best for general quality)
        - **Health_Score_New**: Independent score indicating potential health/hygiene compliance. (Best for health focus)
        - **Rating**: Raw average customer rating.
        """)

    with tab2:
        st.header("In-Depth Data Analysis")
        st.subheader("Location and Cost-Quality Relationship")

        # ----------------------------------------------------
        # Replicating Analysis Tables for Context
        # ----------------------------------------------------

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Top 10 Localities (Restaurant Count)")
            top_localities = df['Locality'].value_counts().nlargest(10)
            st.dataframe(top_localities, use_container_width=True)
        
        with col2:
            st.markdown("##### Correlation Matrix (Key Factors)")
            correlation_cols = ['Rating', 'Votes', 'Cost', 'Trust_Score', 'Health_Score_New', 'Normalized_Mood_Score', 'Final_Robust_Score']
            correlation_matrix = df[correlation_cols].corr()
            # Highlight Robust Score column for easy reading
            st.dataframe(correlation_matrix[['Final_Robust_Score', 'Trust_Score']].style.background_gradient(cmap='RdYlGn'), use_container_width=True)
            st.info("Insight: Robust Score is highly driven by Trust Score and Rating.")

        st.subheader("Visualizations")
        
        # ----------------------------------------------------
        # Replicating Analysis Plots
        # ----------------------------------------------------

        # Plot 1: Top 10 Localities vs. Average Robust Score
        robust_score_by_locality = df[df['Locality'].isin(top_localities.index)].groupby('Locality')['Final_Robust_Score'].mean()
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        robust_score_by_locality.sort_values(ascending=True).plot(kind='barh', ax=ax1, color='skyblue')
        ax1.set_title('A. Average Final Robust Score by Top 10 Locality', fontsize=16)
        ax1.set_xlabel('Average Final Robust Score')
        ax1.set_ylabel('Locality')
        st.pyplot(fig1)
        st.markdown("---")
        
        # Plot 2: Cost Range vs. Average Robust Score
        robust_score_by_cost = df.groupby('Cost_Range')['Final_Robust_Score'].mean().reindex(cost_options)
        
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        robust_score_by_cost.plot(kind='bar', ax=ax2, color=['lightcoral', 'lightgreen', 'gold'])
        ax2.set_title('B. Average Final Robust Score by Cost Range', fontsize=16)
        ax2.set_xlabel('Cost Range')
        ax2.set_ylabel('Average Final Robust Score')
        ax2.tick_params(axis='x', rotation=0)
        st.pyplot(fig2)
        st.markdown("---")

        # Plot 3: Top 10 Cuisines Count
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        top_10_cuisines = df['Cuisine_List'].explode().dropna().value_counts().nlargest(10)
        top_10_cuisines.sort_values(ascending=True).plot(kind='barh', ax=ax3, color='teal')
        ax3.set_title('C. Top 10 Most Frequent Cuisines', fontsize=16)
        ax3.set_xlabel('Number of Restaurants')
        ax3.set_ylabel('Cuisine')
        st.pyplot(fig3)
        st.markdown("---")

        # Plot 4: Rating Distribution
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        sns.histplot(df['Rating'], bins=20, kde=True, color='purple', ax=ax4)
        ax4.set_title('D. Distribution of Restaurant Ratings', fontsize=16)
        ax4.set_xlabel('Rating')
        ax4.set_ylabel('Count')
        st.pyplot(fig4)

# Run the main function
if __name__ == "__main__":
    main()
