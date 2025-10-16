# --- 1. DATA LOADING AND FEATURE ENGINEERING (UPDATED) ---

@st.cache_data
def load_data_and_engineer_features(file_path):
    # Load the dataset
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}. Please check the path.")
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    df['Cuisine_List'] = df['Cuisine'].str.lower().str.replace(' ', '').str.split(',')

    # Trust Score calculation remains the same
    C = df['Rating'].mean()
    m = df['Votes'].quantile(0.8)
    def weighted_rating(df, m, C):
        R = df['Rating']
        v = df['Votes']
        return (v / (v + m) * R) + (m / (v + m) * C)
    df['Trust_Score'] = df.apply(weighted_rating, args=(m, C), axis=1)

    # --- UPDATED Health Score Cuisines (Broader) ---
    # Included Continental and Thai as proxy healthy options for better results
    HEALTHY_CUISINES = ['salad', 'healthfood', 'vegan', 'smoothies', 'glutenfree', 'juices', 'healthyeats', 'health', 'continental', 'thai', 'vietnamese']
    UNHEALTHY_CUISINES = ['desserts', 'icecream', 'fastfood', 'pizza', 'burger', 'bakery', 'streetfood']
    
    def calculate_health_score(cuisine_list):
        score = 0
        for cuisine in cuisine_list:
            if cuisine in HEALTHY_CUISINES: score += 1
            elif cuisine in UNHEALTHY_CUISINES: score -= 1
        return score
    df['Health_Score'] = df['Cuisine_List'].apply(calculate_health_score)

    # Mood Score calculation remains the same
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

    # Tourist Relevance calculation remains the same
    TOURIST_LOCALITIES = ['centraldelhi', 'connaughtplace', 'sarabhanagar', 'sector17']
    def calculate_tourist_relevance(df):
        locality = df['Locality'].str.lower().str.replace(' ', '')
        city = df['City'].str.lower().str.replace(' ', '')
        is_tourist_locality = locality.str.contains('|'.join(TOURIST_LOCALITIES), na=False) | city.str.contains('|'.join(TOURIST_LOCALITIES), na=False)
        return np.where(is_tourist_locality, 1, 0)
    df['Tourist_Relevance'] = calculate_tourist_relevance(df)
    
    return df

# --- 2. MASTER RECOMMENDATION FUNCTION (UPDATED) ---

def recommend_restaurant(df, user_city, user_cuisine, user_mood, user_health, user_tourist, top_n=5):
    filtered_df = df.copy()

    # 1. Apply CITY Filter
    if user_city != "All Cities":
        filtered_df = filtered_df[filtered_df['City'] == user_city]

    # 2. Apply CUISINE Filter
    if user_cuisine != "Any Cuisine":
        cuisine_filter = user_cuisine.lower().replace(' ', '')
        filtered_df = filtered_df[filtered_df['Cuisine_List'].apply(lambda x: cuisine_filter in x)]

    # 3. Apply Mood Filter (Broadened Casual definition)
    if user_mood == "Fine Dining":
        # Target restaurants with high Mood_Score (>= 2 is strongly Fine Dining)
        filtered_df = filtered_df[filtered_df['Mood_Score'] >= 2]
    elif user_mood == "Casual":
        # Target restaurants that are not explicitly fine dining (Mood_Score <= 1)
        filtered_df = filtered_df[filtered_df['Mood_Score'] <= 1]

    # 4. Apply Health Filter
    if user_health == "Healthy":
        # Target restaurants with positive Health_Score
        filtered_df = filtered_df[filtered_df['Health_Score'] >= 1]
    elif user_health == "Indulgent":
        # Target restaurants with negative Health_Score
        filtered_df = filtered_df[filtered_df['Health_Score'] <= -1]

    # 5. Check if any results remain after filtering
    if filtered_df.empty:
        return None

    # --- 6. Final Ranking (Tourist feature is now a SCORE BOOSTER) ---
    
    # Calculate Final Score: Boost Trust_Score for Tourist locations if the checkbox is marked
    if user_tourist:
        filtered_df['Final_Score'] = filtered_df['Trust_Score'] + (filtered_df['Tourist_Relevance'] * 0.2)
    else:
        filtered_df['Final_Score'] = filtered_df['Trust_Score']

    # Final Ranking
    recommendations = filtered_df.sort_values(by='Final_Score', ascending=False)
    
    # Reformat output to show the Trust_Score (the original quality metric)
    return recommendations[['Name', 'City', 'Cuisine', 'Cost', 'Trust_Score']].head(top_n)

# --- 3. STREAMLIT APP LAYOUT (Use the layout from the previous response) ---
# Note: You must ensure the rest of your app.py layout code is present and unchanged.
