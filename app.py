"""
Yelp Gold Layer Dual-Portal RAG System:
1. 👤 Consumer Portal: Continuous Conversational Business Search & Recommendations
2. 💼 Business Owner Portal: Customer Analytics, Sentiment Insights & AI Business Copilot
"""

import sys
import os
import time
from pathlib import Path
import streamlit as st
import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.query_rag import YelpRAGRetriever
import config

# Page Configuration & Aesthetics
st.set_page_config(
    page_title="Yelp RAG Dual Engine & Business Intelligence",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom Styling (Glassmorphism, dark mode, responsive cards)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Gradient Banner */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1e38 50%, #0f172a 100%);
        padding: 24px 32px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        margin-bottom: 24px;
    }
    .main-title {
        color: #f8fafc;
        font-weight: 800;
        font-size: 2.2rem;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 6px;
    }

    /* Portal Badge */
    .portal-badge {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: #ffffff;
        font-weight: 600;
        font-size: 0.8rem;
        padding: 4px 12px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* KPI Cards */
    .kpi-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .kpi-value {
        color: #38bdf8;
        font-size: 1.8rem;
        font-weight: 800;
    }
    .kpi-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Custom Recommendation Card */
    .rec-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 16px;
    }
    .score-chip {
        background: #0284c7;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache DuckDB Database Connection for Fast Aggregations
@st.cache_resource
def get_duckdb_con():
    return duckdb.connect()

# Cache RAG Vector Retriever
@st.cache_resource
def load_retriever():
    return YelpRAGRetriever()

con = get_duckdb_con()
try:
    retriever = load_retriever()
    retriever_ready = True
except Exception as e:
    retriever_ready = False
    retriever_error = str(e)

# Sidebar Role Selector
st.sidebar.markdown("### 🎭 Select Application Mode")
app_mode = st.sidebar.radio(
    "Choose Active Portal:",
    ["👤 Customer / Consumer Portal", "💼 Business Owner Intelligence"],
    index=0
)

st.sidebar.markdown("---")

# Header Render
if "Customer" in app_mode:
    st.markdown("""
    <div class="main-header">
        <div class="main-title">👤 Yelp Customer Assistant & Discovery Engine <span class="portal-badge">Consumer Portal</span></div>
        <div class="main-subtitle">Continuous conversational search to discover top businesses, places, and recommendations based on real Yelp reviews.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="main-header">
        <div class="main-title">💼 Yelp Business Owner Intelligence Dashboard <span class="portal-badge">Owner Copilot</span></div>
        <div class="main-subtitle">Executive review analytics, customer sentiment breakdown, pain point detection & AI copilot for business owners.</div>
    </div>
    """, unsafe_allow_html=True)

if not retriever_ready:
    st.error(f"Error loading RAG Vector Engine: {retriever_error}")
    st.info("Ensure you have run `python scripts/build_rag.py` to construct the local FAISS indices first.")
    st.stop()

# ==============================================================================
# PORTAL 1: CUSTOMER / CONSUMER CONTINUOUS CHAT PORTAL
# ==============================================================================
if "Customer" in app_mode:
    # Sidebar Filters for Customer
    st.sidebar.header("🔍 Search Preferences & Filters")
    cust_city = st.sidebar.text_input("City Filter (e.g. Tampa, Philadelphia, New Orleans)", value="", placeholder="All Cities")
    cust_min_stars = st.sidebar.slider("Minimum Rating (Stars)", 1.0, 5.0, 1.0, 0.5)
    cust_top_k = st.sidebar.slider("Recommendations per Query", 1, 10, 4)

    if st.sidebar.button("🧹 Clear Chat History"):
        st.session_state.customer_messages = []
        st.rerun()

    # Initialize Continuous Chat State
    if "customer_messages" not in st.session_state:
        st.session_state.customer_messages = [
            {
                "role": "assistant",
                "content": "👋 Hi! I am your **Yelp AI Assistant**. Tell me what you're looking for today! (e.g., *'Find top rated Italian places with delicious pasta in Tampa'* or *'Where can I get good coffee with free WiFi?'*)"
            }
        ]

    # Render Continuous Conversation History
    for msg in st.session_state.customer_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Continuous Chat Input Box
    user_input = st.chat_input("Type your request or follow-up question...")

    if user_input:
        # Append User Message
        st.session_state.customer_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Retrieve RAG Context using Semantic Search
        with st.spinner("Searching Gold layer documents & FAISS index..."):
            # Build query with conversational context if follow-up
            search_query = user_input
            if len(st.session_state.customer_messages) > 2:
                # Add previous context summary for continuous multi-turn chat
                last_user_msgs = [m["content"] for m in st.session_state.customer_messages if m["role"] == "user"][-2:]
                search_query = " ".join(last_user_msgs)

            results = retriever.search(
                query_text=search_query,
                doc_type="all",
                city=cust_city.strip() if cust_city.strip() else None,
                min_stars=cust_min_stars if cust_min_stars > 1.0 else None,
                top_k=cust_top_k
            )

        with st.chat_message("assistant"):
            if not results:
                reply = f"I couldn't find matching businesses for **\"{user_input}\"** with the selected filters. Try broadening your city filter or lowering the star rating threshold."
                st.markdown(reply)
                st.session_state.customer_messages.append({"role": "assistant", "content": reply})
            else:
                reply_header = f"Here are the top recommendations based on your request **\"{user_input}\"**:\n\n"
                st.markdown(reply_header)
                full_reply = reply_header

                for idx, res in enumerate(results, 1):
                    doc_type = res["document_type"].upper()
                    biz_name = res.get("business_name", "Unknown Business")
                    city = res.get("city", "")
                    state = res.get("state", "")
                    score = res["score"]
                    rating = res.get("business_rating") if doc_type == "BUSINESS" else res.get("stars")
                    rating_str = f"{rating} ⭐" if rating else "N/A"
                    category = res.get("primary_category", "N/A")
                    
                    excerpt = res["document_text"].replace("\n", "  \n> ")

                    card_md = f"""
---
#### #{idx} **{biz_name}** ({city}, {state})
* **Type:** `{doc_type}` | **Similarity Score:** `{score:.4f}` | **Rating:** {rating_str} | **Category:** `{category}`
> {excerpt[:320]}...
"""
                    st.markdown(card_md)
                    full_reply += card_md

                st.session_state.customer_messages.append({"role": "assistant", "content": full_reply})

# ==============================================================================
# PORTAL 2: BUSINESS OWNER INTELLIGENCE & AI COPILOT
# ==============================================================================
else:
    gold_biz_glob = str(config.GOLD_BUSINESS_PATH / "*.parquet").replace("\\", "/")
    gold_rev_glob = str(config.GOLD_REVIEW_PATH / "*.parquet").replace("\\", "/")

    # Fetch Top Business Names for Owner Selection
    @st.cache_data
    def get_business_names():
        try:
            df_names = con.execute(f"""
                SELECT DISTINCT business_name, city, state, review_count
                FROM read_parquet('{gold_biz_glob}')
                ORDER BY review_count DESC
                LIMIT 500
            """).df()
            df_names["label"] = df_names["business_name"] + " (" + df_names["city"] + ", " + df_names["state"] + ")"
            return df_names
        except Exception as e:
            return pd.DataFrame(columns=["business_name", "city", "state", "review_count", "label"])

    df_biz_list = get_business_names()

    st.sidebar.header("💼 Business Owner Selection")
    if not df_biz_list.empty:
        selected_label = st.sidebar.selectbox(
            "Select Your Business:",
            df_biz_list["label"].tolist(),
            index=0
        )
        selected_biz_name = df_biz_list[df_biz_list["label"] == selected_label]["business_name"].values[0]
    else:
        selected_biz_name = st.sidebar.text_input("Enter Business Name:", value="Acme Oyster House")

    # Fetch Business Analytics Data
    @st.cache_data
    def get_business_metrics(b_name):
        # Business metadata
        biz_info = con.execute(f"""
            SELECT * FROM read_parquet('{gold_biz_glob}')
            WHERE business_name = ?
            LIMIT 1
        """, [b_name]).df()

        # Review statistics & sentiment breakdown
        rev_stats = con.execute(f"""
            SELECT 
                COUNT(*) as total_reviews,
                AVG(stars) as avg_stars,
                COUNT(CASE WHEN sentiment = 'positive' THEN 1 END) as pos_cnt,
                COUNT(CASE WHEN sentiment = 'neutral' THEN 1 END) as neu_cnt,
                COUNT(CASE WHEN sentiment = 'negative' THEN 1 END) as neg_cnt
            FROM read_parquet('{gold_rev_glob}')
            WHERE business_name = ?
        """, [b_name]).fetchone()

        # Top critical complaints (negative reviews)
        neg_reviews = con.execute(f"""
            SELECT stars, review_date, document_text
            FROM read_parquet('{gold_rev_glob}')
            WHERE business_name = ? AND sentiment = 'negative'
            ORDER BY review_date DESC
            LIMIT 5
        """, [b_name]).df()

        # Top positive praise reviews
        pos_reviews = con.execute(f"""
            SELECT stars, review_date, document_text
            FROM read_parquet('{gold_rev_glob}')
            WHERE business_name = ? AND sentiment = 'positive'
            ORDER BY review_date DESC
            LIMIT 5
        """, [b_name]).df()

        return biz_info, rev_stats, neg_reviews, pos_reviews

    biz_info, rev_stats, neg_reviews, pos_reviews = get_business_metrics(selected_biz_name)

    # Executive Overview Header & KPIs
    st.subheader(f"📊 Analytics Executive Summary: {selected_biz_name}")

    if not biz_info.empty:
        b_row = biz_info.iloc[0]
        st.caption(f"📍 **Location:** {b_row.get('address', '')}, {b_row.get('city', '')}, {b_row.get('state', '')} | 🏷️ **Primary Category:** {b_row.get('primary_category', 'N/A')} | ⏰ **Status:** {'Open' if b_row.get('is_open') == 1 else 'Closed'}")

    tot_revs = rev_stats[0] if rev_stats and rev_stats[0] else 0
    avg_stars = rev_stats[1] if rev_stats and rev_stats[1] else 0.0
    pos_cnt = rev_stats[2] if rev_stats and rev_stats[2] else 0
    neu_cnt = rev_stats[3] if rev_stats and rev_stats[3] else 0
    neg_cnt = rev_stats[4] if rev_stats and rev_stats[4] else 0

    pos_pct = (pos_cnt / tot_revs * 100) if tot_revs > 0 else 0
    neg_pct = (neg_cnt / tot_revs * 100) if tot_revs > 0 else 0

    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{avg_stars:.2f} ⭐</div><div class="kpi-label">Average Star Rating</div></div>', unsafe_allow_html=True)
    with col_k2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{tot_revs:,}</div><div class="kpi-label">Total Customer Reviews</div></div>', unsafe_allow_html=True)
    with col_k3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color: #4ade80;">{pos_pct:.1f}%</div><div class="kpi-label">Positive Sentiment ({pos_cnt:,})</div></div>', unsafe_allow_html=True)
    with col_k4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color: #f87171;">{neg_pct:.1f}%</div><div class="kpi-label">Negative Complaints ({neg_cnt:,})</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Owner Dashboard Tabs
    tab_owner_copilot, tab_insights, tab_reviews = st.tabs(["🤖 AI Business Copilot (Chat)", "📈 Sentiment & Pain Point Analysis", "📝 Customer Feedback Archive"])

    # -------------------------------------------------------------
    # TAB 1: AI BUSINESS COPILOT CHATBOT
    # -------------------------------------------------------------
    with tab_owner_copilot:
        st.markdown(f"#### Ask your AI Copilot anything about customer feedback for **{selected_biz_name}**")
        st.caption("Ask questions like: *'What are top customer complaints?'*, *'How can I improve my service rating?'*, or *'What menu items do customers love?'*")

        # Initialize Owner Copilot Chat State
        copilot_key = f"owner_messages_{selected_biz_name}"
        if copilot_key not in st.session_state:
            st.session_state[copilot_key] = [
                {
                    "role": "assistant",
                    "content": f"Hello! I am your AI Business Intelligence Copilot for **{selected_biz_name}**. I have indexed all **{tot_revs:,} customer reviews** for your business. How can I help you improve customer satisfaction today?"
                }
            ]

        # Display Chat History
        for msg in st.session_state[copilot_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat Input Box
        owner_prompt = st.chat_input(f"Ask AI Copilot about {selected_biz_name}...")

        if owner_prompt:
            st.session_state[copilot_key].append({"role": "user", "content": owner_prompt})
            with st.chat_message("user"):
                st.markdown(owner_prompt)

            with st.spinner("Analyzing customer reviews & generating business insights..."):
                # Retrieve specific reviews for this business
                owner_search_query = f"{selected_biz_name} {owner_prompt}"
                biz_rag_results = retriever.search(
                    query_text=owner_search_query,
                    doc_type="review",
                    top_k=5
                )
                
                # Filter specifically for this business if possible
                relevant_revs = [r for r in biz_rag_results if selected_biz_name.lower() in r.get("business_name", "").lower()]
                if not relevant_revs:
                    relevant_revs = biz_rag_results

            with st.chat_message("assistant"):
                copilot_reply = f"### 💡 AI Executive Analysis for **{selected_biz_name}**\n\n"
                copilot_reply += f"Based on analysis of customer feedback for **{selected_biz_name}**:\n\n"
                
                for idx, r in enumerate(relevant_revs[:4], 1):
                    stars = r.get("stars", "N/A")
                    s_label = r.get("sentiment", "N/A").upper()
                    txt = r["document_text"].replace("\n", "  \n> ")
                    copilot_reply += f"""
* **Key Excerpt #{idx} [{stars} ⭐ | {s_label}]**:
> {txt[:280]}...

"""
                copilot_reply += "\n---\n**Actionable Recommendations:**\n"
                if "complaint" in owner_prompt.lower() or "bad" in owner_prompt.lower() or "improve" in owner_prompt.lower():
                    copilot_reply += "1. **Service & Wait Times:** Address peak-hour wait times and streamline table seating workflows.\n2. **Quality Control:** Ensure consistent food temperatures and order accuracy.\n3. **Staff Engagement:** Train floor staff on proactive customer care."
                else:
                    copilot_reply += "1. **Promote Star Items:** Highlight top customer-favored specialties in marketing & social media.\n2. **Customer Loyalty:** Engage with positive reviewers to encourage repeat visits."

                st.markdown(copilot_reply)
                st.session_state[copilot_key].append({"role": "assistant", "content": copilot_reply})

    # -------------------------------------------------------------
    # TAB 2: SENTIMENT & PAIN POINT ANALYSIS
    # -------------------------------------------------------------
    with tab_insights:
        st.markdown(f"#### Sentiment & Pain Point Breakdown for **{selected_biz_name}**")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("##### 🚨 Top Recent Critical Complaints (Negative Reviews)")
            if neg_reviews.empty:
                st.success("No negative complaints found for this business!")
            else:
                for idx, row in neg_reviews.iterrows():
                    st.error(f"⭐ **{row['stars']} Stars** ({row['review_date']})")
                    st.caption(row["document_text"][:300] + "...")
        
        with col_c2:
            st.markdown("##### 🌟 Top Recent Positive Praise (4 & 5 Star Reviews)")
            if pos_reviews.empty:
                st.info("No positive reviews recorded.")
            else:
                for idx, row in pos_reviews.iterrows():
                    st.success(f"⭐ **{row['stars']} Stars** ({row['review_date']})")
                    st.caption(row["document_text"][:300] + "...")

    # -------------------------------------------------------------
    # TAB 3: CUSTOMER FEEDBACK ARCHIVE
    # -------------------------------------------------------------
    with tab_reviews:
        st.markdown(f"#### Full Customer Review Log for **{selected_biz_name}**")
        all_biz_revs = con.execute(f"""
            SELECT review_id, stars, sentiment, review_date, document_text
            FROM read_parquet('{gold_rev_glob}')
            WHERE business_name = ?
            ORDER BY review_date DESC
            LIMIT 50
        """, [selected_biz_name]).df()
        
        if not all_biz_revs.empty:
            st.dataframe(all_biz_revs, use_container_width=True)
        else:
            st.info("No review records found in database.")
