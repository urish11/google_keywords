import streamlit as st
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Google Ads API credentials
CLIENT_ID = st.secrets["google_ads"]["CLIENT_ID"]
CLIENT_SECRET = st.secrets["google_ads"]["CLIENT_SECRET"]
DEVELOPER_TOKEN = st.secrets["google_ads"]["DEVELOPER_TOKEN"]
REFRESH_TOKEN = st.secrets["google_ads"]["REFRESH_TOKEN"]
LOGIN_CUSTOMER_ID = st.secrets["google_ads"]["LOGIN_CUSTOMER_ID"]
CUSTOMER_ID = st.secrets["google_ads"]["CUSTOMER_ID"]

def fetch_keyword_data(keyword, location_id, language_id):
    client = GoogleAdsClient.load_from_dict({
        "developer_token": DEVELOPER_TOKEN,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": REFRESH_TOKEN,
        "login_customer_id": LOGIN_CUSTOMER_ID,
        "use_proto_plus": True
    })

    keyword_plan_idea_service = client.get_service("KeywordPlanIdeaService")

    try:
        request = client.get_type("GenerateKeywordIdeasRequest")
        request.customer_id = CUSTOMER_ID

        geo_target = client.get_type("LocationInfo")
        geo_target.geo_target_constant = f"geoTargetConstants/{location_id}"
        request.geo_target_constants.append(geo_target.geo_target_constant)

        language = client.get_type("LanguageInfo")
        language.language_constant = f"languageConstants/{language_id}"
        request.language = language.language_constant

        keyword_seed = client.get_type("KeywordSeed")
        keyword_seed.keywords.extend([keyword])
        request.keyword_seed = keyword_seed

        response = keyword_plan_idea_service.generate_keyword_ideas(request=request)

        keywords_data = []
        for idea in response.results:
            metrics = idea.keyword_idea_metrics

            keywords_data.append({
                "Keyword": idea.text,
                "Search Volume": metrics.avg_monthly_searches,
                "Competition Index": metrics.competition_index,
                "Low Bid ($)": metrics.low_top_of_page_bid_micros / 1_000_000,
                "High Bid ($)": metrics.high_top_of_page_bid_micros / 1_000_000,
                "Seed Keyword": keyword,
            })

        return pd.DataFrame(keywords_data)

    except GoogleAdsException as ex:
        st.error(f"Error fetching data for keyword '{keyword}': Check your API credentials and parameters.")
        return pd.DataFrame()

def calculate_quantitative_index(df, weight_volume, weight_competition, weight_bids):
    # Add an Average Bid column
    df["Average Bid"] = (df["Low Bid ($)"] + df["High Bid ($)"]) / 2

    # Normalize columns
    columns_to_normalize = ["Search Volume", "Competition Index", "Average Bid"]
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[columns_to_normalize])
    normalized_df = pd.DataFrame(normalized_data, columns=[f"Normalized {col}" for col in columns_to_normalize])

    # Calculate the index
    df["Quantitative Index"] = (
        normalized_df["Normalized Search Volume"] * weight_volume +
        normalized_df["Normalized Competition Index"] * weight_competition +
        normalized_df["Normalized Average Bid"] * weight_bids
    )

    return df.sort_values(by="Quantitative Index", ascending=False)

# Streamlit App
st.title("Google Ads Keyword Ideas with Custom Weights")

# User input for keywords
keywords_input = st.text_area("Enter a list of keywords (one per line):")

# User input for location and language
location_id = st.text_input("Enter Location ID (e.g., 2840 for US):", "2840")
language_id = st.text_input("Enter Language ID (e.g., 1000 for English):", "1000")

# User input for weights
st.write("### Set Weights for the Index Calculation")
weight_volume = st.slider("Weight for Search Volume", 0.0, 1.0, 0.5)
weight_competition = st.slider("Weight for Competition Index", 0.0, 1.0, 0.3)
weight_bids = st.slider("Weight for Bid Average", 0.0, 1.0, 0.2)

if st.button("Fetch Keyword Ideas"):
    with st.spinner("Fetching data..."):
        keywords = [kw.strip() for kw in keywords_input.splitlines() if kw.strip()]
        if not keywords:
            st.error("Please enter at least one keyword.")
        else:
            try:
                location_id = int(location_id)
                language_id = int(language_id)
                all_data = pd.DataFrame()
                for keyword in keywords:
                    data = fetch_keyword_data(keyword, location_id, language_id)
                    all_data = pd.concat([all_data, data], ignore_index=True)

                if not all_data.empty:
                    # Calculate Quantitative Index with user-defined weights
                    all_data = calculate_quantitative_index(all_data, weight_volume, weight_competition, weight_bids)

                    # Display the table
                    sort_by = st.selectbox("Sort by:", all_data.columns, index=1)
                    all_data = all_data.sort_values(by=sort_by, ascending=False)
                    st.dataframe(all_data)

                    # Download as CSV
                    csv = all_data.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name="keyword_ideas.csv",
                        mime="text/csv",
                    )
                else:
                    st.write("No data available.")
            except ValueError:
                st.error("Please enter valid numeric values for Location ID and Language ID.")
