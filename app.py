import streamlit as st
st.set_page_config(layout="wide")
import random
import google.ads.googleads
import requests
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
from google import genai
import json
from st_aggrid import AgGrid, GridOptionsBuilder
from datetime import datetime
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import DBSCAN
# import nltk
import numpy as np
from collections import Counter
import math
import anthropic
# # Ensure NLTK dependencies are downloaded
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')
 
# Google Ads API credentials (use st.secrets for sensitive data)
CLIENT_ID = st.secrets["google_ads"]["CLIENT_ID"]
CLIENT_SECRET = st.secrets["google_ads"]["CLIENT_SECRET"]
DEVELOPER_TOKEN = st.secrets["google_ads"]["DEVELOPER_TOKEN"]
REFRESH_TOKEN = st.secrets["google_ads"]["REFRESH_TOKEN"]
LOGIN_CUSTOMER_ID = st.secrets["google_ads"]["LOGIN_CUSTOMER_ID"]
CUSTOMER_ID = st.secrets["google_ads"]["CUSTOMER_ID"]

GPT_API_KEY = st.secrets["google_ads"]["GPT_API_KEY"]

GEMINI_API_KEYS= st.secrets["google_ads"]["GEMINI_API_KEY"]
agg_df= None

# Full list of locations (Country Name -> Location ID)
locations = {
    2840: "United States", 2004: "Afghanistan", 2008: "Albania",2010: "Antarctica",2012: "Algeria",2016: "American Samoa",2020: "Andorra",2024: "Angola",2028: "Antigua and Barbuda",2031: "Azerbaijan",2032: "Argentina",2036: "Australia",2040: "Austria",2044: "The Bahamas",2048: "Bahrain",2050: "Bangladesh",2051: "Armenia",2052: "Barbados",2056: "Belgium",2064: "Bhutan",2068: "Bolivia",2070: "Bosnia and Herzegovina",2072: "Botswana",2076: "Brazil",2084: "Belize",2090: "Solomon Islands",2096: "Brunei",2100: "Bulgaria",2104: "Myanmar (Burma)",2108: "Burundi",2112: "Belarus",2116: "Cambodia",2120: "Cameroon",2124: "Canada",2132: "Cabo Verde",2140: "Central African Republic",2144: "Sri Lanka",2148: "Chad",2152: "Chile",2156: "China",2162: "Christmas Island",2166: "Cocos (Keeling) Islands",2170: "Colombia",2174: "Comoros",2178: "Republic of the Congo",2180: "Democratic Republic of the Congo",2184: "Cook Islands",2188: "Costa Rica",2191: "Croatia",2196: "Cyprus",2203: "Czechia",2204: "Benin",2208: "Denmark",2212: "Dominica",2214: "Dominican Republic",2218: "Ecuador",2222: "El Salvador",2226: "Equatorial Guinea",2231: "Ethiopia",2232: "Eritrea",2233: "Estonia",2239: "South Georgia and the South Sandwich Islands",2242: "Fiji",2246: "Finland",2250: "France",2258: "French Polynesia",2260: "French Southern and Antarctic Lands",2262: "Djibouti",2266: "Gabon",2268: "Georgia",2270: "The Gambia",2276: "Germany",2288: "Ghana",2296: "Kiribati",2300: "Greece",2308: "Grenada",2316: "Guam",2320: "Guatemala",2324: "Guinea",2328: "Guyana",2332: "Haiti",2334: "Heard Island and McDonald Islands",2336: "Vatican City",2340: "Honduras",2348: "Hungary",2352: "Iceland",2356: "India",2360: "Indonesia",2368: "Iraq",2372: "Ireland",2376: "Israel",2380: "Italy",2384: "Cote d'Ivoire",2388: "Jamaica",2392: "Japan",2398: "Kazakhstan",2400: "Jordan",2404: "Kenya",2410: "South Korea",2414: "Kuwait",2417: "Kyrgyzstan",2418: "Laos",2422: "Lebanon",2426: "Lesotho",2428: "Latvia",2430: "Liberia",2434: "Libya",2438: "Liechtenstein",2440: "Lithuania",2442: "Luxembourg",2450: "Madagascar",2454: "Malawi",2458: "Malaysia",2462: "Maldives",2466: "Mali",2470: "Malta",2478: "Mauritania",2480: "Mauritius",2484: "Mexico",2492: "Monaco",2496: "Mongolia",2498: "Moldova",2499: "Montenegro",2504: "Morocco",2508: "Mozambique",2512: "Oman",2516: "Namibia",2520: "Nauru",2524: "Nepal",2528: "Netherlands",2531: "Curacao",2534: "Sint Maarten",2535: "Caribbean Netherlands",2540: "New Caledonia",2548: "Vanuatu",2554: "New Zealand",2558: "Nicaragua",2562: "Niger",2566: "Nigeria",2570: "Niue",2574: "Norfolk Island",2578: "Norway",2580: "Northern Mariana Islands",2581: "United States Minor Outlying Islands",2583: "Micronesia",2584: "Marshall Islands",2585: "Palau",2586: "Pakistan",2591: "Panama",2598: "Papua New Guinea",2600: "Paraguay",2604: "Peru",2608: "Philippines",2612: "Pitcairn Islands",2616: "Poland",2620: "Portugal",2624: "Guinea-Bissau",2626: "Timor-Leste",2634: "Qatar",2642: "Romania",2643: "Russia",2646: "Rwanda",2652: "Saint Barthelemy",2654: "Saint Helena, Ascension and Tristan da Cunha",2659: "Saint Kitts and Nevis",2662: "Saint Lucia",2663: "Saint Martin",2666: "Saint Pierre and Miquelon",2670: "Saint Vincent and the Grenadines",2674: "San Marino",2678: "Sao Tome and Principe",2682: "Saudi Arabia",2686: "Senegal",2688: "Serbia",2690: "Seychelles",2694: "Sierra Leone",2702: "Singapore",2703: "Slovakia",2704: "Vietnam",2705: "Slovenia",2706: "Somalia",2710: "South Africa",2716: "Zimbabwe",2724: "Spain",2728: "South Sudan",2736: "Sudan",2740: "Suriname",2748: "Eswatini",2752: "Sweden",2756: "Switzerland",2762: "Tajikistan",2764: "Thailand",2768: "Togo",2772: "Tokelau",2776: "Tonga",2780: "Trinidad and Tobago",2784: "United Arab Emirates",2788: "Tunisia",2792: "Turkiye",2795: "Turkmenistan",2798: "Tuvalu",2800: "Uganda",2804: "Ukraine",2807: "North Macedonia",2818: "Egypt",2826: "United Kingdom",2831: "Guernsey",2832: "Jersey",2833: "Isle of Man",2834: "Tanzania",2854: "Burkina Faso",2858: "Uruguay",2860: "Uzbekistan",2862: "Venezuela",2876: "Wallis and Futuna",2882: "Samoa",2887: "Yemen",2894: "Zambia"

}

# Full list of languages (Language Name -> Language ID)
languages = {
    1000: "English", 1001: "German", 1002: "French", 1003: "Spanish", 1004: "Italian",
    1005: "Japanese", 1009: "Danish", 1010: "Dutch", 1011: "Finnish", 1012: "Korean",
    1013: "Norwegian", 1014: "Portuguese", 1015: "Swedish", 1017: "Chinese (Simplified)",
    1018: "Chinese (Traditional)", 1019: "Arabic", 1020: "Bulgarian", 1021: "Czech",
    1022: "Greek", 1023: "Hindi", 1024: "Hungarian", 1025: "Indonesian", 1026: "Icelandic",
    1027: "Hebrew", 1028: "Latvian", 1029: "Lithuanian", 1030: "Polish", 1031: "Russian",
    1032: "Romanian", 1033: "Slovak", 1034: "Slovenian", 1035: "Serbian", 1036: "Ukrainian",
    1037: "Turkish", 1038: "Catalan", 1039: "Croatian", 1040: "Vietnamese", 1041: "Urdu",
    1042: "Filipino", 1043: "Estonian", 1044: "Thai", 1056: "Bengali", 1064: "Persian",
    1072: "Gujarati", 1086: "Kannada", 1098: "Malayalam", 1101: "Marathi", 1102: "Malay",
    1110: "Punjabi", 1130: "Tamil", 1131: "Telugu"
}


client = GoogleAdsClient.load_from_dict({
    "developer_token": DEVELOPER_TOKEN,
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "refresh_token": REFRESH_TOKEN,
    "login_customer_id": LOGIN_CUSTOMER_ID,
    "use_proto_plus": True
})
month_enum = client.enums.MonthOfYearEnum

def chatGPT(prompt, model="gpt-4o", temperature=1.0) :
    st.write("Generating image description...")
    headers = {
        'Authorization': f'Bearer {GPT_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': model,
        'temperature': temperature,
        'messages': [{'role': 'user', 'content': prompt}]
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
    content = response.json()['choices'][0]['message']['content'].strip()
    return  content


def claude(prompt , model = "claude-sonnet-4-20250514", temperature=0.87 , is_thinking = False, max_retries = 10): # claude-3-7-sonnet-latest
    # if is_pd_policy_global : prompt +=   PREDICT_POLICY
    tries = 0
    st.text(f"Using model: {model}")
    # st.text(prompt)
    while tries < max_retries:
        try:
        
        
        
            client = anthropic.Anthropic(
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
            api_key=st.secrets["google_ads"]["ANTHROPIC_API_KEY"])
        
            if is_thinking == False:
                    
                message = client.messages.create(
                    
                model=model,
                max_tokens=20000,
                temperature=temperature,
                
                top_p= 0.8,

                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
                return message.content[0].text
            if is_thinking == True:
                message = client.messages.create(
                    
                model=model,
                max_tokens=20000,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                thinking = { "type": "enabled",
                "budget_tokens": 16000}
            )
                return message.content[1].text
        
        
        
            print(message)
            return message.content[0].text

        except Exception as e:
            st.text(e)
            tries += 1 
            time.sleep(5)

def month_number_to_google_enum(month_number, month_enum):
    return {
        1: month_enum.JANUARY,
        2: month_enum.FEBRUARY,
        3: month_enum.MARCH,
        4: month_enum.APRIL,
        5: month_enum.MAY,
        6: month_enum.JUNE,
        7: month_enum.JULY,
        8: month_enum.AUGUST,
        9: month_enum.SEPTEMBER,
        10: month_enum.OCTOBER,
        11: month_enum.NOVEMBER,
        12: month_enum.DECEMBER,
    }[month_number]

def fetch_keyword_data(keyword, location_id, language_id , network):
    try:

        
        # client = GoogleAdsClient.load_from_dict({
        #     "developer_token": DEVELOPER_TOKEN,
        #     "client_id": CLIENT_ID,
        #     "client_secret": CLIENT_SECRET,
        #     "refresh_token": REFRESH_TOKEN,
        #     "login_customer_id": LOGIN_CUSTOMER_ID,
        #     "use_proto_plus": True
        # })

        keyword_plan_idea_service = client.get_service("KeywordPlanIdeaService")

        request = client.get_type("GenerateKeywordIdeasRequest")
        request.customer_id = CUSTOMER_ID

        geo_target = client.get_type("LocationInfo")
        geo_target.geo_target_constant = f"geoTargetConstants/{location_id}"
        request.geo_target_constants.append(geo_target.geo_target_constant)

        language = client.get_type("LanguageInfo")
        language.language_constant = f"languageConstants/{language_id}"
        request.language = language.language_constant



        start_month_enum = month_number_to_google_enum(start_month, month_enum)
        end_month_enum = month_number_to_google_enum(end_month, month_enum)

        year_month_range = request.historical_metrics_options.year_month_range
        year_month_range.start.year = start_year
        year_month_range.start.month = start_month_enum
        year_month_range.end.year = end_year
        year_month_range.end.month = end_month_enum

        # network = client.get_type("KeywordPlanNetwork")
        # network.network_constant = f"KeywordPlanNetwork/GOOGLE_SEARCH"
        if network == "GOOGLE_SEARCH_AND_PARTNERS":
            request.keyword_plan_network = client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH_AND_PARTNERS 
        elif network == "GOOGLE_SEARCH":
            request.keyword_plan_network = client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH
        
        keyword_seed = client.get_type("KeywordSeed")
        keyword_seed.keywords.extend(keyword)
        request.keyword_seed = keyword_seed

        # url_seed = client.get_type("UrlSeed")
        # url_seed.url = "https://searchlabz.com/motorcycle-payment-plans-with-no-credit-check-en/"
        # request.url_seed= url_seed
 
        response = keyword_plan_idea_service.generate_keyword_ideas(request=request)
        # st.text(str(response))
        keywords_data = []
        ils_usd = 3.6
        for idea in response.results:
            metrics = idea.keyword_idea_metrics
            if metrics.avg_monthly_searches > 0 and (metrics.low_top_of_page_bid_micros > 0 ):  # Exclude rows with Search Volume == 0
                search_volume_sum = sum( m.monthly_searches for m in metrics.monthly_search_volumes if (m.year > start_year or (m.year == start_year and m.month >= start_month_enum)) 
                                        and (m.year < end_year or (m.year == end_year and m.month <= end_month_enum)) )
                keywords_data.append({
                    "Keyword": idea.text,
                    # "Search Volume": metrics.monthly_search_volumes[-1].monthly_searches,
                    "Search Volume": search_volume_sum,
                    "Competition Index": round(metrics.competition_index, 2),
                    "Low Bid ($)": round(metrics.low_top_of_page_bid_micros / 1_000_000 / ils_usd, 2),
                    "High Bid ($)": round(metrics.high_top_of_page_bid_micros / 1_000_000 / ils_usd, 2),
                    "Network" : network
                })

        return pd.DataFrame(keywords_data)

    except Exception as ex:
        st.error(f"Error fetching data for keyword '{keyword}': {ex}")
        return pd.DataFrame()
    except:
        time.sleep(1)

def gemini_text_lib(prompt, model="gemini-2.5-flash-preview-04-17",max_retries=5): # Using a stable model  
    tries = 0
    while tries < max_retries:
        
        st.text(f"Gemini working.. {model} trial {tries+1}")
        """ Calls Gemini API, handling potential list of keys """
        if not GEMINI_API_KEYS:
            st.error("Gemini API keys not available.")
            return None
    
        # If multiple keys, choose one randomly; otherwise use the configured one (if single) or the first.
        selected_key = random.choice(GEMINI_API_KEYS)
    
        client = genai.Client(api_key=selected_key)
    
    
        try:
            response = client.models.generate_content(
                model=model, contents=  prompt
            )
            st.text(str(response))
    
            return response.text
        except Exception as e:
            st.text('gemini_text_lib error ' + str(e)) 
            time.sleep(15)
            tries += 1
    
    return None
def get_network_delta(df):
    # Step 1: Normalize the keywords
    df['Keyword'] = df['Keyword'].str.strip().str.lower()

    # Step 2: Create pivot with search volumes per network
    pivot = df.pivot_table(index='Keyword', columns='Network', values='Search Volume', aggfunc='first')

    # Step 3: Compute the difference for keywords that exist in both networks
    pivot['Search Volume Diff'] = pivot.get('GOOGLE_SEARCH_AND_PARTNERS') - pivot.get('GOOGLE_SEARCH')

    # Step 4: Merge the calculated diff back only for GOOGLE_SEARCH_AND_PARTNERS rows
    df['Search Volume Diff'] = None  # start with None
    for idx, row in df.iterrows():
        if row['Network'] == 'GOOGLE_SEARCH_AND_PARTNERS':
            kw = row['Keyword']
            if kw in pivot.index:
                df.at[idx, 'Search Volume Diff'] = pivot.at[kw, 'Search Volume Diff']

    return df



def calculate_quantitative_index(df, weight_volume, weight_competition, weight_bids):
    df["Average Bid"] = (df["Low Bid ($)"] + df["High Bid ($)"]) / 2

    columns_to_normalize = ["Search Volume", "Competition Index", "Low Bid ($)"]
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[columns_to_normalize])
    normalized_df = pd.DataFrame(normalized_data, columns=[f"Normalized {col}" for col in columns_to_normalize])

    df["Quantitative Index"] = (
        normalized_df["Normalized Search Volume"] * weight_volume +
        normalized_df["Normalized Competition Index"] * weight_competition +
        np.log10(normalized_df["Normalized Low Bid ($)"] +1  ) * weight_bids
    )

    df["Quantitative Index"] = df["Quantitative Index"].round(4)
    df["Average Bid"] = df["Average Bid"].round(2)

    return df.sort_values(by="Quantitative Index", ascending=False)

def dynamic_keyword_clustering(keywords, ngram_range=(1, 3), eps=0.5, min_samples=2):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words='english')
    X = vectorizer.fit_transform(keywords)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', algorithm='brute').fit(X)

    cluster_labels = clustering.labels_

    return pd.DataFrame({
        "Keyword": keywords,
        "Cluster": cluster_labels
    })

def get_representative_phrase(keywords_series):
    stop_words = set(stopwords.words('english')).union({"best", "accurate", "with", "and", "device", "digital"})  # Add custom stopwords
    
    # Convert Series to a list and drop any null or empty strings
    keywords = [kw for kw in keywords_series.dropna().tolist() if kw.strip()]
    
    if not keywords:  # If no valid keywords remain
        return ""
    
    # Tokenize all keywords in the group and remove stopwords
    tokenized_keywords = [
        [word for word in word_tokenize(keyword.lower()) if word not in stop_words]
        for keyword in keywords
    ]
    
    # Flatten the list of tokens
    all_tokens = [token for tokens in tokenized_keywords for token in tokens]
    
    # Check if all_tokens is empty
    if not all_tokens:
        return min(keywords, key=len)  # Fallback to shortest keyword if no tokens remain
    
    # Identify the most common words in the group
    most_common_words = Counter(all_tokens).most_common()
    
    # Limit phrase length to top 3 most frequent tokens
    concise_phrase = " ".join([word for word, count in most_common_words[:3]])
    
    # Default to the shortest keyword if no common words are found
    if not concise_phrase:
        return min(keywords, key=len)
    
    return concise_phrase




def aggregate_by_cluster(data, cluster_data):
    data = data.merge(cluster_data, on="Keyword", how="left")

    def weighted_average(x):
        return np.average(x, weights=data.loc[x.index, 'Search Volume'])

    # Sort keywords by volume within each cluster
    sorted_keywords = (
        data.groupby("Cluster")
        .apply(lambda x: ", ".join(
            x.sort_values(by="Search Volume", ascending=False)["Keyword"]
        ))
        .reset_index(name="Cluster Keywords")
    )

    # Get representative phrase for each cluster by ensuring uniqueness
    keywords_agg = cluster_data.groupby('Cluster')['Keyword'].apply(lambda x: get_representative_phrase(x)).reset_index()

    # Calculate aggregated metrics
    metrics_agg = data.groupby('Cluster').agg({
        'Search Volume': 'sum',
        'Competition Index': weighted_average,
        'Low Bid ($)': 'mean',
        'High Bid ($)': 'mean',
        'Quantitative Index': weighted_average
    }).reset_index()

    # Merge aggregated metrics with representative keywords
    aggregated_data = pd.merge(metrics_agg, keywords_agg, on='Cluster', how='left')

    # Merge sorted keywords into aggregated data
    aggregated_data = pd.merge(aggregated_data, sorted_keywords, on='Cluster', how='left')

    # Rename for output clarity
    aggregated_data.rename(columns={
        'Keyword': 'Key Phrase',
        'Search Volume': 'Total_Search_Volume',
        'Competition Index': 'Weighted Avg Competition Index',
        'Low Bid ($)': 'Avg_Low_Bid',
        'High Bid ($)': 'Avg_High_Bid',
        'Quantitative Index': 'Weighted Avg Quantitative Index'
    }, inplace=True)

    # Filter out noise
    aggregated_data = aggregated_data[aggregated_data["Cluster"] != -1]

    # Round numerical columns
    for col in ["Total_Search_Volume", "Weighted Avg Competition Index", 
                "Avg_Low_Bid", "Avg_High_Bid", "Weighted Avg Quantitative Index"]:
        aggregated_data[col] = aggregated_data[col].round(2)


    desired_order = [
    'Cluster', 
    'Key Phrase', 
    'Total_Search_Volume', 
    'Weighted Avg Competition Index', 
    'Avg_Low_Bid', 
    'Avg_High_Bid', 
    'Weighted Avg Quantitative Index', 
    'Cluster Keywords'
        ]
    aggregated_data = aggregated_data[desired_order]

    return aggregated_data



# Streamlit App

st.title("Google Ads Keyword Ideas with Quantitative Index")

selected_location = st.selectbox("Select Location:", options=list(locations.keys()), format_func=lambda x: locations[x])
selected_language = st.selectbox("Select Language:", options=list(languages.keys()), format_func=lambda x: languages[x])
keywords_input = st.text_area("Enter a list of keywords (one per line):",height=300,)
 
st.write("### Set Weights for Quantitative Index")
weight_volume = st.slider("Weight for Search Volume", 0.0, 1.0, 0.5)
weight_competition = st.slider("Weight for Competition Index", 0.0, 1.0, 0.3)
weight_bids = st.slider("Weight for Average Bid", 0.0, 1.0, 0.2)

enable_aggregation = st.checkbox("Enable Dynamic Keyword Aggregation", value=True)
enable_gpt_kws = st.checkbox("Add KWs via chatGPT?", value=False)
new_but_diff_kws = st.checkbox("New but different KWs via Claude?", value=False)
if enable_gpt_kws:
    count_gpt_kws = st.number_input('How Many GPT KWs?',value = 20)

if new_but_diff_kws:
    new_but_diff_kws_count = st.number_input('How Many new KWs factor?',value = 3)
years = list(range(2019, 2026))
months = list(range(1, 13))

col1, col2 ,_,_,_= st.columns(5)

with col1:
    start_year = st.selectbox("Start Year", years, index=years.index(datetime.now().date().year))
    start_month = st.selectbox("Start Month", months, index=datetime.now().date().month-3)

with col2:
    end_year = st.selectbox("End Year", years, index=years.index(2025))
    end_month = st.selectbox("End Month", months, index=datetime.now().date().month-1)



if st.button("Fetch Keyword Ideas"):
    all_data=None
    st.session_state["all_data"] = None
    with st.spinner("Fetching data..."):
        keywords = [kw.strip() for kw in keywords_input.splitlines() if kw.strip()]

        if enable_gpt_kws:
            gpt_kws = chatGPT(f"write more {str(count_gpt_kws)} diverse and divergent (CONCISE AS POSSIBLE)! keywords (not nesseacrly containg original) , return JUST THE PLAIN TXT the new keywords each spereted with  no bullit points no list of numbers just the kws spereated by \n for: {keywords_input} in the same language as input")
            keywords = keywords + gpt_kws.split("\n")
        elif new_but_diff_kws : 
            keywords = claude(f"""give me new simillar  but DIFFERENT new ideas not synonyms concise no duplicates kws for search arb with high intent and high CPC like \n {chr(92) + 'n'.join(keywords)} \n\n 
             Return same format, no intros just pure data\n {round(new_but_diff_kws_count*len(keywords) ,-1)} rows""").split('\n')
            with st.expander("New KWs from Claude"):
                st.text('\n'.join(keywords))
             
   
            
        

        if not keywords:
            st.error("Please enter at least one keyword.")
        else:
            with st.expander("AdPlanner Log"):
                all_data = pd.DataFrame()
                st.text("going to google")
                n_of_chunks =len(keywords)// 20 + 1
                if len(keywords) < 20 : n_of_chunks=1
                # n_of_chunks= len(keywords)

               
                chunks = np.array_split(np.array(keywords),n_of_chunks  )

                for chunk_n, chunk in enumerate(chunks):
                    st.text(f"chunk {chunk_n} out of {n_of_chunks}")
                    chunk = list(chunk.tolist())
                    st.text(chunk)
                    for network in ["GOOGLE_SEARCH_AND_PARTNERS", "GOOGLE_SEARCH"]:
                        data = fetch_keyword_data(chunk, selected_location, selected_language,network)
                        time.sleep(1)

                        all_data = pd.concat([all_data, data], ignore_index=True)

                    st.text(f"Total len {len(all_data)}")
                st.text("done")
                # st.text(all_data)
                all_data = get_network_delta(all_data)

                if not all_data.empty:
                    all_data = calculate_quantitative_index(all_data, weight_volume, weight_competition, weight_bids)
                    all_data = all_data.drop_duplicates()
                    all_data = all_data[all_data["Keyword"].str.count(" ") >=2]
                    all_data =all_data.sort_values(by="Search Volume Diff")
                    st.session_state["all_data"] = all_data

if "all_data" in st.session_state:
    all_data = st.session_state["all_data"]

    # Display Original Table
    st.write("### Interactive Table (Original Data)")
    csv = all_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "data.csv", "text/csv") 
    gb = GridOptionsBuilder.from_dataframe(all_data)
    gb.configure_pagination(enabled=True,paginationPageSize=50)
    gb.configure_default_column(filterable=True, sortable=True, editable=True)
    gb.configure_column("Network", filter=True)
    gb.configure_column("Keyword", filter=True)
    gb.configure_grid_options(enableRangeSelection=True,rowSelection="multiple")  # Enable range selection
    gb.configure_grid_options(pagination=True, paginationPageSize=50, paginationAutoPageSize=False, paginationMode="client")
    gb.configure_column("Search Volume Diff", type="numberColumnFilter", precision=0)

#     gb.configure_column(
#     "sel",  # A dummy column name, AgGrid will create it
#     headerCheckboxSelection=True,  # Checkbox in the header for "select all"
#     checkboxSelection=True,       # Checkbox for each row
#     pinned="left",                # Pin to the left for visibility
#     lockPosition=True,            # Prevent moving it
#     suppressMenu=True,            # No menu for this column
#     width=5                      # Adjust width
# )
    col1, _ = st.columns([1, 7])  
    with col1:
        page_size = st.selectbox("Rows per page", [10, 20, 50, 100, 200, 500, 1000], index=3,)

    gb.configure_grid_options(clipboard=True)  # Enable clipboard copy
    gb.configure_selection(selection_mode='multiple', use_checkbox=True)

    gb.configure_column("Search Volume", sort="desc")
    gb.configure_column(
    field="Keyword",
    headerCheckboxSelection=True,     # Select all checkbox in header
    checkboxSelection=True            # Checkboxes per row
)


    grid_options = gb.build()
    grid_options['paginationPageSize'] = page_size
    grid_options['pagination'] = True
    grid_options['paginationAutoPageSize'] = False
    grid_options['domLayout'] = 'autoHeight'

    grid_options['paginationPageSizeOptions'] = [10, 20, 50, 100, 200,400,500,600,700,800,900,1000,2000]  # 👈 Here are your presets
    grid_options['suppressPaginationPanel'] = False  # Must be False to show the dropdown

    grid_response = AgGrid(all_data, gridOptions=grid_options, height=800, width=700, theme="streamlit",update_mode='SELECTION_CHANGED')


    

if st.button("Proccess!"):
    st.session_state["trigger_process"] = True
    st.session_state["selected_rows"] = grid_response['selected_rows']

if st.session_state.get("trigger_process") and "selected_rows" in st.session_state:
    st.session_state["trigger_process"] = False

    selected_df = pd.DataFrame(grid_response['selected_rows']).reset_index()
    if selected_df.empty:
        st.warning("Please select at least one row.")
    else:
        st.dataframe(selected_df)

        subset = pd.DataFrame({'name': selected_df['Keyword']})
        st.text(subset)

        prompt= """Please go over the following search arbitrage ideas, i want u to group these kws to remove repeating ones, like if u see rent to own vehicles no deposit AND cars rent to own no deposit group them into a concise 1 term like :'rent to own vehicles no deposit'
                                
                                group close keywords that would yield same search results on google like :["rent to own homes near me","rent to own homes","cheap rent to own houses near me"] are 1 group for example (ehrn write new groupd idea text dont use special chars !)
                                im going to provied you with table 2 col : idea , indecies

                                i want u to group the ideas and reurn json of idea and list of indecies,
                                [{idea:'idea1...', indices:[list_of_indices]},{idea:'idea2...', indices:[list_of_indices]}...]

                                no intros no extra JUST the json


                                """ + subset.to_csv()
        with st.status("Processing prompt..."):
            st.text(prompt)
            group_res = gemini_text_lib(prompt).replace("```json","").replace("```","")
            st.text(group_res)

        try:
            grouped_ideas = json.loads(group_res)
        except json.JSONDecodeError as e:
            st.error("Failed to parse JSON from Gemini output.")
            st.stop()

        agg_results = []
        for group in grouped_ideas:
            indices = group.get("indices", [])
            idea = group.get("idea", "")
            group_rows = selected_df.iloc[indices]
            kws = "\n".join(group_rows["Keyword"].tolist())
            total_volume = group_rows["Search Volume"].sum() or 1

            def weighted_avg(col):
                return (group_rows[col] * group_rows["Search Volume"]).sum() / total_volume

            agg_results.append({
                "Grouped Idea": idea,
                "Count": len(group_rows),
                "Total Search Volume": group_rows["Search Volume"].sum(),
                "Weighted Avg Competition Index": weighted_avg("Competition Index"),
                "Weighted Avg Low Bid ($)": weighted_avg("Low Bid ($)"),
                "Weighted Avg High Bid ($)": weighted_avg("High Bid ($)"),
                "Weighted Avg Quantitative Index": weighted_avg("Quantitative Index"),
                "Sum Search Volume Diff": group_rows.get("Search Volume Diff", pd.Series()).sum(),
                "KWs" : kws
            })

        agg_df = pd.DataFrame(agg_results)
        st.write("### Aggregated Grouped Ideas (Weighted by Search Volume)")
        st.dataframe(agg_df)


    # if selected_rows_data:
    #     # AgGrid(selected_rows_data,gridOptions=grid_options, height=800, width=700, theme="streamlit")
    #     selected_df = pd.DataFrame(selected_rows_data)

    #     st.dataframe(selected_df)


    # if enable_aggregation:
    #     # Perform Dynamic Clustering
    #     cluster_data = dynamic_keyword_clustering(all_data["Keyword"].tolist(), ngram_range=(2, 3), eps=0.6, min_samples=2)

    #     # Aggregate Data by Clusters
    #     aggregated_table = aggregate_by_cluster(all_data, cluster_data)

    #     # Display Aggregated Table
    #     st.write("### Aggregated Table (By Key Phrase)")
    #     gb = GridOptionsBuilder.from_dataframe(aggregated_table)
    #     gb.configure_pagination(enabled=True,paginationPageSize=100)
    #     gb.configure_default_column(filterable=True, sortable=True, editable=False)
    #     gb.configure_column("Keyword", filter=True)
    #     gb.configure_grid_options(enableRangeSelection=True)  # Enable range selection
    #     gb.configure_grid_options(pagination=True, paginationPageSize=50, paginationAutoPageSize=False, paginationMode="client")

    #     gb.configure_grid_options(clipboard=True)  # Enable clipboard copy
    #     gb.configure_column("Total_Search_Volume", sort="desc")



    #     grid_options = gb.build()
    #     AgGrid(aggregated_table, gridOptions=grid_options, height=800, width=700, theme="streamlit")

    # Download Button for Original Table
    csv = all_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Original Data as CSV",
        data=csv,
        file_name="keyword_ideas.csv",
        mime="text/csv",
    )
    # # csv_cluster = aggregated_table.to_csv(index=False).encode("utf-8")
    # st.download_button(
    #     label="Download aggregated_table  CSV",
    #     data=csv,
    #     file_name="keyword_ideas_agg.csv",
    #     mime="text/csv",
    # )
