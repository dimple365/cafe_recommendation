import streamlit as st
import pandas as pd
import numpy as np

# --- 1. DATA LOADING AND FEATURE ENGINEERING (Your Core Logic - UNCHANGED) ---

@st.cache_data
def load_data_and_engineer_features(file_path):
    # Load the dataset
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}. Please check the path.")
        return pd.DataFrame() # Return empty DataFrame on error

    df = pd.read_csv(file_path) # Assume file load succeeds from here
    df['Cuisine_List'] = df['Cuisine'].str.lower().str.replace(' ', '').str.split(',')

    # Trust Score
    C = df['Rating'].mean()
    m = df['Votes'].quantile(0.8)
    def weighted_rating(df, m, C):
        R = df['Rating']
        v = df['Votes']
        return (v / (v + m) * R) + (m / (v + m) * C)
    df['Trust_Score'] = df.apply(weighted_rating, args=(m, C), axis=1)

    # Health Score
    HEALTHY_CUISINES = ['salad', 'healthfood', 'vegan', 'smoothies', 'glutenfree', 'juices', 'healthyeats', 'health']
    UNHEALTHY_CUISINES = ['desserts', 'icecream', 'fastfood', 'pizza', 'burger', 'bakery', 'streetfood']
    def calculate_health_score(cuisine_list):
        score = 0
        for cuisine in cuisine_list:
            if cuisine in HEALTHY_CUISINES: score += 1
            elif cuisine in UNHEALTHY_CUISINES: score -= 1
        return score
    df['Health_Score'] = df['Cuisine_List'].apply(calculate_health_score)

    # Mood Score
    median_cost = df['Cost'].median()
    FINE_DINING_CUISINES = ['continental', 'italian', 'seafood', 'european', 'french', 'mediterranean']
    CASUAL_CUISINES = ['cafe', 'fingerfood', 'northindian', 'chinese', 'mughlai']
    def calculate_mood_score(row):
        score = 0
        for cuisine in row['Cuisine_List']:
            if cuisine in FINE_DINING_CUISINES: score += 1
            elif cuisine in CASUAL_CUISINES: score -= 1
        if row['Cost'] > median_cost: score += 1
        elif row['Cost'] < median_cost: score -= 1
        return score
    df['Mood_Score'] = df.apply(calculate_mood_score, axis=1)

    # Tourist Relevance
    TOURIST_LOCALITIES = ['centraldelhi', 'connaughtplace', 'sarabhanagar', 'sector17']
    def calculate_tourist_relevance(df):
        locality = df['Locality'].str.lower().str.replace(' ', '')
        city = df['City'].str.lower().str.replace(' ', '')
        is_tourist_locality = locality.str.contains('|'.join(TOURIST_LOCALITIES), na=False) | city.str.contains('|'.join(TOURIST_LOCALITIES), na=False)
        return np.where(is_tourist_locality, 1, 0)
    df['Tourist_Relevance'] = calculate_tourist_relevance(df)
    
    return df

# --- 2. MASTER RECOMMENDATION FUNCTION (The Engine - MODIFIED for City Filter) ---

def recommend_restaurant(df, user_city, user_mood, user_health, user_tourist, top_n=5):
    filtered_df = df.copy()

    # 1. Apply CITY Filter (New primary location filter)
    if user_city != "All Cities":
        filtered_df = filtered_df[filtered_df['City'] == user_city]

    # 2. Apply Mood Filter
    if user_mood == "Fine Dining":
        filtered_df = filtered_df[filtered_df['Mood_Score'] >= 2]
    elif user_mood == "Casual":
        filtered_df = filtered_df[filtered_df['Mood_Score'] <= 0]

    # 3. Apply Health Filter
    if user_health == "Healthy":
        filtered_df = filtered_df[filtered_df['Health_Score'] >= 1]
    elif user_health == "Indulgent":
        filtered_df = filtered_df[filtered_df['Health_Score'] <= -1]

    # 4. Apply Tourist Filter (Now filters LOCATIONS *within* the selected City)
    if user_tourist:
        filtered_df = filtered_df[filtered_df['Tourist_Relevance'] == 1]

    if filtered_df.empty:
        return None

    # 5. Final Ranking by Trustworthiness (Trust_Score)
    recommendations = filtered_df.sort_values(by='Trust_Score', ascending=False)
    
    return recommendations[['Name', 'City', 'Cuisine', 'Cost', 'Trust_Score']].head(top_n)

# --- 3. STREAMLIT APP LAYOUT (UPDATED) ---

# Load the data once
df_restaurants = load_data_and_engineer_features("restaurants.csv")

st.title("🍽️ Multi-Factor Restaurant Recommender")
st.markdown("Use the filters below to get recommendations based on **Health**, **Mood**, and **Location**, ranked by a robust **Trust Score**.")

if not df_restaurants.empty:
    
    # Get unique cities for the dropdown
    city_options = ["All Cities"] + sorted(df_restaurants['City'].unique().tolist())
    
    # Sidebar for Filters (User Input)
    st.sidebar.header("Your Preferences")
    
    # 1. City Location Filter (NEW)
    city_choice = st.sidebar.selectbox(
        "Select City:",
        city_options,
        index=0
    )

    # 2. Tourist Filter (Refined Location)
    tourist_choice = st.sidebar.checkbox("Focus on Tourist/Major Localities?", value=False)
    
    st.sidebar.markdown("---")
    
    # 3. Mood Filter
    mood_choice = st.sidebar.selectbox(
        "Mood/Ambiance:",
        ["Any", "Casual", "Fine Dining"],
        index=0
    )

    # 4. Health Filter
    health_choice = st.sidebar.selectbox(
        "Health Focus:",
        ["Any", "Healthy", "Indulgent"],
        index=0
    )
    
    st.sidebar.markdown("---")

    # Run the recommendation engine
    results = recommend_restaurant(
        df_restaurants, 
        user_city=city_choice, # Pass the new argument
        user_mood=mood_choice, 
        user_health=health_choice, 
        user_tourist=tourist_choice
    )

    st.header("Top Recommendations")
    
    if results is not None:
        st.dataframe(results, hide_index=True, use_container_width=True)

        # Trustworthiness Feature Highlight
        st.markdown("---")
        st.subheader("Trustworthiness: Reliable Recommendations Only")
        max_trust = results['Trust_Score'].iloc[0]
        st.metric(label="Highest Trust Score in Results", value=f"{max_trust:.2f} ⭐", delta="Reliability Vetted")
        st.caption("The Trust Score is a weighted rating that ensures recommendations are based on high-volume, reliable votes.")
    else:
        st.warning(f"Sorry, no restaurants matched ALL your criteria in **{city_choice}**. Try broadening your search.")
