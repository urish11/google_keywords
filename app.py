import streamlit as st
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
from st_aggrid import AgGrid, GridOptionsBuilder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk

# Ensure NLTK dependencies are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Streamlit App
st.title("Google Ads Keyword Ideas with Quantitative Index and Aggregation")

# Google Ads API credentials (use st.secrets for sensitive data)
CLIENT_ID = st.secrets["google_ads"]["CLIENT_ID"]
CLIENT_SECRET = st.secrets["google_ads"]["CLIENT_SECRET"]
DEVELOPER_TOKEN = st.secrets["google_ads"]["DEVELOPER_TOKEN"]
REFRESH_TOKEN = st.secrets["google_ads"]["REFRESH_TOKEN"]
LOGIN_CUSTOMER_ID = st.secrets["google_ads"]["LOGIN_CUSTOMER_ID"]
CUSTOMER_ID = st.secrets["google_ads"]["CUSTOMER_ID"]

# Full list of locations and languages (example dictionary)
locations = {
     2004: "Afghanistan", 2008: "Albania",2010: "Antarctica",2012: "Algeria",2016: "American Samoa",2020: "Andorra",2024: "Angola",2028: "Antigua and Barbuda",2031: "Azerbaijan",2032: "Argentina",2036: "Australia",2040: "Austria",2044: "The Bahamas",2048: "Bahrain",2050: "Bangladesh",2051: "Armenia",2052: "Barbados",2056: "Belgium",2064: "Bhutan",2068: "Bolivia",2070: "Bosnia and Herzegovina",2072: "Botswana",2076: "Brazil",2084: "Belize",2090: "Solomon Islands",2096: "Brunei",2100: "Bulgaria",2104: "Myanmar (Burma)",2108: "Burundi",2112: "Belarus",2116: "Cambodia",2120: "Cameroon",2124: "Canada",2132: "Cabo Verde",2140: "Central African Republic",2144: "Sri Lanka",2148: "Chad",2152: "Chile",2156: "China",2162: "Christmas Island",2166: "Cocos (Keeling) Islands",2170: "Colombia",2174: "Comoros",2178: "Republic of the Congo",2180: "Democratic Republic of the Congo",2184: "Cook Islands",2188: "Costa Rica",2191: "Croatia",2196: "Cyprus",2203: "Czechia",2204: "Benin",2208: "Denmark",2212: "Dominica",2214: "Dominican Republic",2218: "Ecuador",2222: "El Salvador",2226: "Equatorial Guinea",2231: "Ethiopia",2232: "Eritrea",2233: "Estonia",2239: "South Georgia and the South Sandwich Islands",2242: "Fiji",2246: "Finland",2250: "France",2258: "French Polynesia",2260: "French Southern and Antarctic Lands",2262: "Djibouti",2266: "Gabon",2268: "Georgia",2270: "The Gambia",2276: "Germany",2288: "Ghana",2296: "Kiribati",2300: "Greece",2308: "Grenada",2316: "Guam",2320: "Guatemala",2324: "Guinea",2328: "Guyana",2332: "Haiti",2334: "Heard Island and McDonald Islands",2336: "Vatican City",2340: "Honduras",2348: "Hungary",2352: "Iceland",2356: "India",2360: "Indonesia",2368: "Iraq",2372: "Ireland",2376: "Israel",2380: "Italy",2384: "Cote d'Ivoire",2388: "Jamaica",2392: "Japan",2398: "Kazakhstan",2400: "Jordan",2404: "Kenya",2410: "South Korea",2414: "Kuwait",2417: "Kyrgyzstan",2418: "Laos",2422: "Lebanon",2426: "Lesotho",2428: "Latvia",2430: "Liberia",2434: "Libya",2438: "Liechtenstein",2440: "Lithuania",2442: "Luxembourg",2450: "Madagascar",2454: "Malawi",2458: "Malaysia",2462: "Maldives",2466: "Mali",2470: "Malta",2478: "Mauritania",2480: "Mauritius",2484: "Mexico",2492: "Monaco",2496: "Mongolia",2498: "Moldova",2499: "Montenegro",2504: "Morocco",2508: "Mozambique",2512: "Oman",2516: "Namibia",2520: "Nauru",2524: "Nepal",2528: "Netherlands",2531: "Curacao",2534: "Sint Maarten",2535: "Caribbean Netherlands",2540: "New Caledonia",2548: "Vanuatu",2554: "New Zealand",2558: "Nicaragua",2562: "Niger",2566: "Nigeria",2570: "Niue",2574: "Norfolk Island",2578: "Norway",2580: "Northern Mariana Islands",2581: "United States Minor Outlying Islands",2583: "Micronesia",2584: "Marshall Islands",2585: "Palau",2586: "Pakistan",2591: "Panama",2598: "Papua New Guinea",2600: "Paraguay",2604: "Peru",2608: "Philippines",2612: "Pitcairn Islands",2616: "Poland",2620: "Portugal",2624: "Guinea-Bissau",2626: "Timor-Leste",2634: "Qatar",2642: "Romania",2643: "Russia",2646: "Rwanda",2652: "Saint Barthelemy",2654: "Saint Helena, Ascension and Tristan da Cunha",2659: "Saint Kitts and Nevis",2662: "Saint Lucia",2663: "Saint Martin",2666: "Saint Pierre and Miquelon",2670: "Saint Vincent and the Grenadines",2674: "San Marino",2678: "Sao Tome and Principe",2682: "Saudi Arabia",2686: "Senegal",2688: "Serbia",2690: "Seychelles",2694: "Sierra Leone",2702: "Singapore",2703: "Slovakia",2704: "Vietnam",2705: "Slovenia",2706: "Somalia",2710: "South Africa",2716: "Zimbabwe",2724: "Spain",2728: "South Sudan",2736: "Sudan",2740: "Suriname",2748: "Eswatini",2752: "Sweden",2756: "Switzerland",2762: "Tajikistan",2764: "Thailand",2768: "Togo",2772: "Tokelau",2776: "Tonga",2780: "Trinidad and Tobago",2784: "United Arab Emirates",2788: "Tunisia",2792: "Turkiye",2795: "Turkmenistan",2798: "Tuvalu",2800: "Uganda",2804: "Ukraine",2807: "North Macedonia",2818: "Egypt",2826: "United Kingdom",2831: "Guernsey",2832: "Jersey",2833: "Isle of Man",2834: "Tanzania",2840: "United States",2854: "Burkina Faso",2858: "Uruguay",2860: "Uzbekistan",2862: "Venezuela",2876: "Wallis and Futuna",2882: "Samoa",2887: "Yemen",2894: "Zambia"

}

# Full list of languages (Language Name -> Language ID)
languages = {
    1000: "English", 1001: "Spanish", 1002: "French", 1003: "German", 1004: "Italian",
    1005: "Dutch", 1006: "Portuguese", 1007: "Russian", 1008: "Japanese", 1009: "Korean",
    1010: "Chinese (Simplified)", 1011: "Chinese (Traditional)", 1012: "Arabic",
    1013: "Hindi", 1014: "Bengali", 1015: "Turkish", 1016: "Vietnamese"
    # Add the full list from the provided data here...
}
selected_location = st.selectbox("Select Location:", options=list(locations.keys()), format_func=lambda x: locations[x])
selected_language = st.selectbox("Select Language:", options=list(languages.keys()), format_func=lambda x: languages[x])
keywords_input = st.text_area("Enter a list of keywords (one per line):")

st.write("### Set Weights for Quantitative Index")
weight_volume = st.slider("Weight for Search Volume", 0.0, 1.0, 0.5)
weight_competition = st.slider("Weight for Competition Index", 0.0, 1.0, 0.3)
weight_bids = st.slider("Weight for Average Bid", 0.0, 1.0, 0.2)

enable_aggregation = st.checkbox("Enable Dynamic Keyword Aggregation", value=True)

if st.button("Fetch Keyword Ideas"):
    with st.spinner("Fetching data..."):
        keywords = [kw.strip() for kw in keywords_input.splitlines() if kw.strip()]
        if not keywords:
            st.error("Please enter at least one keyword.")
        else:
            all_data = pd.DataFrame()
            for keyword in keywords:
                # Simulated data for simplicity
                data = pd.DataFrame({
                    "Keyword Phrase": [keyword],
                    "Search Volume": [1000 * len(keyword)],
                    "Competition Index": [0.5],
                    "Low Bid ($)": [0.1],
                    "High Bid ($)": [0.2],
                })
                all_data = pd.concat([all_data, data], ignore_index=True)

            # Calculate Quantitative Index
            all_data["Average Bid"] = (all_data["Low Bid ($)"] + all_data["High Bid ($)"]) / 2
            columns_to_normalize = ["Search Volume", "Competition Index", "Average Bid"]
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(all_data[columns_to_normalize])
            normalized_df = pd.DataFrame(normalized_data, columns=[f"Normalized {col}" for col in columns_to_normalize])
            all_data["Quantitative Index"] = (
                normalized_df["Normalized Search Volume"] * weight_volume +
                normalized_df["Normalized Competition Index"] * weight_competition +
                normalized_df["Normalized Average Bid"] * weight_bids
            )

            st.session_state["all_data"] = all_data

if "all_data" in st.session_state:
    all_data = st.session_state["all_data"]

    # Original Table
    st.write("### Interactive Table")
    gb = GridOptionsBuilder.from_dataframe(all_data)
    gb.configure_pagination(paginationPageSize=100)
    gb.configure_default_column(filterable=True, sortable=True, editable=False)
    grid_options = gb.build()
    AgGrid(all_data, gridOptions=grid_options, height=500, width=1000, theme="streamlit")

    if enable_aggregation:
        # Dynamic Keyword Aggregation
        stop_words = set(stopwords.words("english"))
        all_data["Tokens"] = all_data["Keyword Phrase"].apply(lambda x: [
            word.lower() for word in word_tokenize(x) if word.lower() not in stop_words and word.isalnum()
        ])
        all_tokens = [token for tokens in all_data["Tokens"] for token in tokens]
        keyword_counts = Counter(all_tokens)

        def map_to_keyword(tokens):
            for token in tokens:
                if token in keyword_counts:
                    return token
            return "other"

        all_data["Key Keyword"] = all_data["Tokens"].apply(map_to_keyword)
        aggregated_df = all_data.groupby("Key Keyword").agg(
            Total_Search_Volume=("Search Volume", "sum"),
            Avg_Competition_Index=("Competition Index", "mean"),
            Total_Quantitative_Index=("Quantitative Index", "sum")
        ).reset_index()

        # Display Aggregated Table
        st.write("### Aggregated Table")
        gb = GridOptionsBuilder.from_dataframe(aggregated_df)
        gb.configure_pagination(paginationPageSize=100)
        gb.configure_default_column(filterable=True, sortable=True, editable=False)
        grid_options = gb.build()
        AgGrid(aggregated_df, gridOptions=grid_options, height=500, width=1000, theme="streamlit")

    # Download Button
    csv = all_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="keyword_ideas.csv",
        mime="text/csv",
    )
