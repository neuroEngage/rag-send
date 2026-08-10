"""
Yelp Gold Layer Dual-Portal RAG Application (Yelp Cookbook Design System)

Features:
- Official Yelp "Cookbook" Color Palette & Typography (Poppins & Open Sans)
- Dual-Portal Architecture:
  1. 👤 Consumer Search & Discovery Portal
  2. 💼 Business Owner Intelligence Dashboard & AI Copilot
- 66% / 33% Yelp Business Grid Layout
- Positive Praise Reviews FIRST, Critical Complaints SECOND
- High-Contrast, Legible KPI Cards & Review Cards
- Business-Scoped RAG Retrieval for 100% Business-Specific Evidence
- AI Copilot with Question Summary & Actionable Recommendations Summary
"""

import sys
import os
import time
from pathlib import Path
import streamlit as st
import duckdb
import pandas as pd
from dotenv import load_dotenv

# Add parent project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

import config
from scripts.query_rag import YelpRAGRetriever
from rag.rag_pipeline import answer_question

# ==============================================================================
# PAGE CONFIGURATION & STYLING (YELP COOKBOOK DESIGN SYSTEM)
# ==============================================================================
st.set_page_config(
    page_title="Yelp Business Intelligence & RAG Engine",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS implementing Yelp's Cookbook Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@0,400;0,600;0,700;1,400&family=Poppins:wght@500;600;700;800&display=swap');

    /* Design System Tokens */
    :root {
        --yelp-red: #FA4848;
        --yelp-red-dark: #D71616;
        --yelp-red-light: #FFECEC;
        --yelp-teal: #0396BC;
        --yelp-teal-dark: #007692;
        --yelp-teal-light: #D9F6FD;
        --yelp-green: #029E6A;
        --text-primary: #2D2E2F;
        --text-secondary: #6B6D6F;
        --text-disabled: #898A8B;
        --bg-page: #F7F7F7;
        --bg-card: #FFFFFF;
        --border-color: #E3E3E3;
        --star-gold: #FA4848;
        --star-gray: #C8C9CA;
    }

    /* Global Typography Reset */
    html, body, [class*="css"] {
        font-family: 'Open Sans', sans-serif;
        color: var(--text-primary);
        background-color: var(--bg-page);
    }

    h1, h2, h3, h4, h5, h6, .poppins-font {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700;
        color: var(--text-primary);
    }

    /* Streamlit Main Container Adjustment */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    /* Header / Navbar */
    .yelp-navbar {
        background-color: #FFFFFF;
        border-bottom: 2px solid var(--border-color);
        padding: 14px 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .yelp-logo {
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: 2.1rem;
        color: var(--yelp-red);
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: -0.5px;
    }
    .yelp-badge {
        background-color: var(--yelp-red-light);
        color: var(--yelp-red-dark);
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 0.8rem;
        padding: 4px 12px;
        border-radius: 16px;
        border: 1px solid rgba(250, 72, 72, 0.2);
    }

    /* KPI Cards (High Contrast & Legible) */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card-white {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .kpi-number {
        font-family: 'Poppins', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .kpi-label {
        font-family: 'Open Sans', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 6px;
    }

    /* Yelp Star Ribbon */
    .star-ribbon {
        color: var(--star-gold);
        font-size: 1.2rem;
        letter-spacing: 2px;
    }

    /* Badges & Selection Pills */
    .yelp-pill {
        display: inline-block;
        background-color: #F0F0F0;
        color: var(--text-primary);
        font-family: 'Open Sans', sans-serif;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .yelp-pill-teal {
        background-color: #FFFFFF;
        color: var(--yelp-teal);
        border: 1.5px solid var(--yelp-teal);
        font-family: 'Poppins', sans-serif;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 4px 14px;
        border-radius: 20px;
        display: inline-block;
        margin: 4px;
        cursor: pointer;
    }

    /* Review Cards */
    .review-card-pos {
        background-color: #FFFFFF;
        border-left: 5px solid var(--yelp-green);
        border-top: 1px solid var(--border-color);
        border-right: 1px solid var(--border-color);
        border-bottom: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .review-card-neg {
        background-color: var(--yelp-red-light);
        border-left: 5px solid var(--yelp-red);
        border-top: 1px solid var(--border-color);
        border-right: 1px solid var(--border-color);
        border-bottom: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }

    .review-author {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        color: var(--text-primary);
    }
    .review-date {
        font-size: 0.8rem;
        color: var(--text-secondary);
    }
    .review-text {
        font-size: 0.92rem;
        color: var(--text-primary);
        margin-top: 8px;
        line-height: 1.5;
    }

    /* Sidebar Component Box */
    .sidebar-widget {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
</style>
""", unsafe_allow_html=True)


# Cache DuckDB Database Connection
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

# ==============================================================================
# SIDEBAR CONTROLS & APPLICATION MODE SELECTION
# ==============================================================================
st.sidebar.markdown("### 🛑 Select Application Mode")
app_mode = st.sidebar.radio(
    "Choose Active Portal:",
    ["👤 Consumer / Search Portal", "💼 Business Owner Intelligence"],
    index=0
)

st.sidebar.markdown("---")

# Render Yelp Navbar Header
if "Consumer" in app_mode:
    st.markdown("""<div class="yelp-navbar">
<div class="yelp-logo">
<span style="color: #FA4848;">yelp</span><span style="font-size: 1.3rem; color: #2D2E2F;">⚙️</span> 
<span style="font-size: 1.2rem; font-weight:600; color: #2D2E2F; margin-left: 6px;">Consumer Search Portal</span>
</div>
<div>
<span class="yelp-badge">Consumer RAG Engine</span>
</div>
</div>""", unsafe_allow_html=True)
else:
    st.markdown("""<div class="yelp-navbar">
<div class="yelp-logo">
<span style="color: #FA4848;">yelp</span><span style="font-size: 1.3rem; color: #0396BC;">💼</span> 
<span style="font-size: 1.2rem; font-weight:600; color: #2D2E2F; margin-left: 6px;">Business Intelligence Engine</span>
</div>
<div>
<span class="yelp-badge" style="background-color: #D9F6FD; color: #007692; border-color: rgba(3, 150, 188, 0.3);">Owner Intelligence</span>
</div>
</div>""", unsafe_allow_html=True)

if not retriever_ready:
    st.error(f"Error loading RAG Vector Engine: {retriever_error}")
    st.info("Ensure FAISS indices are generated in `models/faiss_index`.")
    st.stop()


def render_stars(rating):
    if not rating:
        return "☆☆☆☆☆"
    full_stars = int(round(rating))
    return "★" * full_stars + "☆" * (5 - full_stars)


# ==============================================================================
# PORTAL 1: CUSTOMER / CONSUMER PORTAL
# ==============================================================================
if "Consumer" in app_mode:
    st.sidebar.header("🔍 Search Preferences")
    cust_city = st.sidebar.text_input("City Filter", value="", placeholder="e.g. Philadelphia, Tampa, New Orleans")
    cust_min_stars = st.sidebar.slider("Minimum Rating (Stars)", 1.0, 5.0, 1.0, 0.5)
    cust_top_k = st.sidebar.slider("Results per Query", 1, 10, 4)

    if st.sidebar.button("🧹 Clear Chat History"):
        st.session_state.customer_messages = []
        st.rerun()

    if "customer_messages" not in st.session_state:
        st.session_state.customer_messages = [
            {
                "role": "assistant",
                "content": "👋 **Welcome to Yelp Search!** Tell me what you are looking for today (e.g. *'Best authentic Italian pizza with good wine in Philadelphia'* or *'Where can I get good coffee with free WiFi?'*)"
            }
        ]

    for msg in st.session_state.customer_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    user_input = st.chat_input("Ask Yelp AI Assistant for business recommendations...")

    if user_input:
        st.session_state.customer_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("🔍 Searching Yelp index & generating AI answer..."):
            search_query = user_input
            if len(st.session_state.customer_messages) > 2:
                last_user_msgs = [m["content"] for m in st.session_state.customer_messages if m["role"] == "user"][-2:]
                search_query = " ".join(last_user_msgs)

            rag_output = answer_question(
                question=search_query,
                doc_type="all",
                city=cust_city.strip() if cust_city.strip() else None,
                min_stars=cust_min_stars if cust_min_stars > 1.0 else None,
                top_k=cust_top_k,
            )
            ai_answer = rag_output["answer"]
            results = rag_output["sources"]

        with st.chat_message("assistant"):
            if not results:
                reply = f"No businesses matching **\"{user_input}\"** were found. Try adjusting your city filter or star rating threshold."
                st.markdown(reply)
                st.session_state.customer_messages.append({"role": "assistant", "content": reply})
            else:
                cards_html = []
                for idx, res in enumerate(results, 1):
                    doc_type = res["document_type"].upper()
                    biz_name = res.get("business_name", "Unknown Business")
                    city = res.get("city", "")
                    state = res.get("state", "")
                    score = res["score"]
                    rating = res.get("business_rating") if doc_type == "BUSINESS" else res.get("stars")
                    rating_val = float(rating) if rating else 0.0
                    stars_html = render_stars(rating_val)
                    category = res.get("primary_category", "Restaurant")
                    sentiment = res.get("sentiment", "")
                    excerpt = res["document_text"].replace("\n", " ")

                    sentiment_color = "#029E6A" if sentiment == "positive" else "#D71616" if sentiment == "negative" else "#6B6D6F"
                    sentiment_bg = "#E8F8F2" if sentiment == "positive" else "#FFECEC" if sentiment == "negative" else "#F0F0F0"

                    card_md = f"""<div style="background:#FFFFFF; border:1px solid #E3E3E3; border-radius:12px; padding:18px; margin-bottom:16px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
<div style="font-family:'Poppins',sans-serif; font-size:1.1rem; font-weight:700; color:#2D2E2F;">
<span style="color:#FA4848;">#{idx}</span> {biz_name} <span style="font-size:0.85rem; font-weight:400; color:#6B6D6F;">({city}, {state})</span>
</div>
<div style="margin-top:6px; margin-bottom:10px;">
<span style="color:#FA4848; font-size:1.0rem; font-weight:bold;">{stars_html}</span> 
<span style="font-weight:700; color:#2D2E2F; font-size:0.9rem; margin-left:4px;">{rating_val:.1f}</span>
<span class="yelp-pill" style="margin-left:8px;">{category}</span>
{f'<span class="yelp-pill" style="background:{sentiment_bg}; color:{sentiment_color};">{sentiment.capitalize()}</span>' if sentiment else ''}
<span class="yelp-pill" style="background:#D9F6FD; color:#007692;">sim={score:.3f}</span>
</div>
<div style="font-size:0.9rem; color:#2D2E2F; line-height:1.5; background:#F7F7F7; padding:12px; border-radius:8px; border-left:4px solid #0396BC;">
"{excerpt[:300]}..."
</div>
</div>"""
                    cards_html.append(card_md)

                full_reply = f"""<div style="background:linear-gradient(135deg,#FA4848 0%,#D71616 100%); border-radius:14px; padding:20px 24px; margin-bottom:20px; box-shadow:0 4px 16px rgba(250,72,72,0.25);">
<div style="font-family:'Poppins',sans-serif; font-size:0.8rem; font-weight:700; color:rgba(255,255,255,0.8); text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;">🤖 AI Business Insight</div>
</div>

{ai_answer}

---
### 📄 Evidence — Top Results for **"{user_input}"**

""" + "\n\n".join(cards_html)

                st.markdown(full_reply, unsafe_allow_html=True)
                st.session_state.customer_messages.append({"role": "assistant", "content": full_reply})


# ==============================================================================
# PORTAL 2: BUSINESS OWNER INTELLIGENCE & COPILOT (COOKBOOK GRID LAYOUT)
# ==============================================================================
else:
    gold_biz_glob = str(config.GOLD_BUSINESS_PATH / "*.parquet").replace("\\", "/")
    gold_rev_glob = str(config.GOLD_REVIEW_PATH / "*.parquet").replace("\\", "/")

    @st.cache_data
    def search_businesses(search_term=""):
        try:
            if search_term and search_term.strip():
                term = f"%{search_term.strip()}%"
                df_names = con.execute(f"""
                    SELECT business_id, business_name, city, state, primary_category, review_count
                    FROM read_parquet('{gold_biz_glob}')
                    WHERE business_name ILIKE ? OR city ILIKE ? OR primary_category ILIKE ?
                    ORDER BY review_count DESC
                    LIMIT 300
                """, [term, term, term]).df()
            else:
                df_names = con.execute(f"""
                    SELECT business_id, business_name, city, state, primary_category, review_count
                    FROM read_parquet('{gold_biz_glob}')
                    ORDER BY review_count DESC
                    LIMIT 300
                """).df()
            
            if not df_names.empty:
                df_names["label"] = df_names["business_name"] + " (" + df_names["city"] + ", " + df_names["state"] + ")"
            return df_names
        except Exception as e:
            return pd.DataFrame(columns=["business_id", "business_name", "city", "state", "primary_category", "review_count", "label"])

    st.sidebar.header("💼 Business Selection")
    biz_search_input = st.sidebar.text_input(
        "🔎 Search Any Business:",
        value="",
        placeholder="Type name, city, or category...",
        key="biz_search_input_key"
    )

    df_biz_list = search_businesses(biz_search_input)

    if not df_biz_list.empty:
        selected_label = st.sidebar.selectbox(
            "Choose Active Business Location:",
            df_biz_list["label"].tolist(),
            index=0,
            key="biz_selectbox_key",
            help="Select business location to instantly refresh dashboard."
        )
        selected_row = df_biz_list[df_biz_list["label"] == selected_label].iloc[0]
        selected_biz_id = selected_row["business_id"]
        selected_biz_name = selected_row["business_name"]
        selected_city = selected_row["city"]
        selected_state = selected_row["state"]
    else:
        st.sidebar.warning(f"No business found matching '{biz_search_input}'.")
        selected_biz_id = None
        selected_biz_name = "Acme Oyster House"
        selected_city = "New Orleans"
        selected_state = "LA"

    @st.cache_data
    def get_business_metrics(b_id, fallback_name="Acme Oyster House"):
        if b_id:
            biz_info = con.execute(f"""
                SELECT * FROM read_parquet('{gold_biz_glob}')
                WHERE business_id = ?
                LIMIT 1
            """, [b_id]).df()

            rev_stats = con.execute(f"""
                SELECT 
                    COUNT(*) as total_reviews,
                    AVG(stars) as avg_stars,
                    COUNT(CASE WHEN sentiment = 'positive' THEN 1 END) as pos_cnt,
                    COUNT(CASE WHEN sentiment = 'neutral' THEN 1 END) as neu_cnt,
                    COUNT(CASE WHEN sentiment = 'negative' THEN 1 END) as neg_cnt
                FROM read_parquet('{gold_rev_glob}')
                WHERE business_id = ?
            """, [b_id]).fetchone()

            pos_reviews = con.execute(f"""
                SELECT stars, review_date, document_text
                FROM read_parquet('{gold_rev_glob}')
                WHERE business_id = ? AND sentiment = 'positive'
                ORDER BY review_date DESC
                LIMIT 6
            """, [b_id]).df()

            neg_reviews = con.execute(f"""
                SELECT stars, review_date, document_text
                FROM read_parquet('{gold_rev_glob}')
                WHERE business_id = ? AND sentiment = 'negative'
                ORDER BY review_date DESC
                LIMIT 6
            """, [b_id]).df()
        else:
            biz_info = con.execute(f"""
                SELECT * FROM read_parquet('{gold_biz_glob}')
                WHERE business_name = ?
                LIMIT 1
            """, [fallback_name]).df()

            rev_stats = con.execute(f"""
                SELECT 
                    COUNT(*) as total_reviews,
                    AVG(stars) as avg_stars,
                    COUNT(CASE WHEN sentiment = 'positive' THEN 1 END) as pos_cnt,
                    COUNT(CASE WHEN sentiment = 'neutral' THEN 1 END) as neu_cnt,
                    COUNT(CASE WHEN sentiment = 'negative' THEN 1 END) as neg_cnt
                FROM read_parquet('{gold_rev_glob}')
                WHERE business_name = ?
            """, [fallback_name]).fetchone()

            pos_reviews = con.execute(f"""
                SELECT stars, review_date, document_text
                FROM read_parquet('{gold_rev_glob}')
                WHERE business_name = ? AND sentiment = 'positive'
                ORDER BY review_date DESC
                LIMIT 6
            """, [fallback_name]).df()

            neg_reviews = con.execute(f"""
                SELECT stars, review_date, document_text
                FROM read_parquet('{gold_rev_glob}')
                WHERE business_name = ? AND sentiment = 'negative'
                ORDER BY review_date DESC
                LIMIT 6
            """, [fallback_name]).df()

        return biz_info, rev_stats, pos_reviews, neg_reviews

    biz_info, rev_stats, pos_reviews, neg_reviews = get_business_metrics(selected_biz_id, selected_biz_name)

    # HERO SECTION
    b_row = biz_info.iloc[0] if not biz_info.empty else {}
    selected_biz_name = b_row.get("business_name", selected_biz_name)
    city_st = f"{b_row.get('city', selected_city)}, {b_row.get('state', selected_state)}"
    category = b_row.get("primary_category", "Restaurant")
    address = b_row.get("address", "Main Street")
    is_open = b_row.get("is_open", 1) == 1

    tot_revs = rev_stats[0] if rev_stats and rev_stats[0] else 0
    avg_stars = rev_stats[1] if rev_stats and rev_stats[1] else 0.0
    pos_cnt = rev_stats[2] if rev_stats and rev_stats[2] else 0
    neg_cnt = rev_stats[4] if rev_stats and rev_stats[4] else 0

    pos_pct = (pos_cnt / tot_revs * 100) if tot_revs > 0 else 0.0
    neg_pct = (neg_cnt / tot_revs * 100) if tot_revs > 0 else 0.0

    hero_html = f"""<div style="background:#FFFFFF; border:1px solid #E3E3E3; border-radius:16px; padding:28px; margin-bottom:24px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
<div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap;">
<div>
<h1 style="font-family:'Poppins',sans-serif; font-size:2.4rem; margin:0; color:#2D2E2F;">{selected_biz_name}</h1>
<div style="margin-top:8px; display:flex; align-items:center; gap:10px;">
<span style="color:#FA4848; font-size:1.3rem;">{render_stars(avg_stars)}</span>
<span style="font-family:'Poppins',sans-serif; font-weight:700; font-size:1.1rem; color:#2D2E2F;">{avg_stars:.2f}</span>
<span style="color:#6B6D6F; font-size:0.95rem;">({tot_revs:,} customer reviews)</span>
</div>
<div style="margin-top:12px;">
<span class="yelp-pill" style="background:#F0F0F0; font-weight:700;">$$</span>
<span class="yelp-pill" style="background:#F0F0F0; font-weight:600;">{category}</span>
<span class="yelp-pill" style="background:#E8F8F2; color:#029E6A; font-weight:700;">{'Open Now' if is_open else 'Closed'}</span>
<span style="color:#6B6D6F; font-size:0.9rem; margin-left:8px;">📍 {address}, {city_st}</span>
</div>
</div>
</div>
</div>"""
    st.markdown(hero_html, unsafe_allow_html=True)

    # EXECUTIVE KPI CARDS
    kpi_html = f"""<div class="kpi-container">
<div class="kpi-card-white">
<div class="kpi-number" style="color: #2D2E2F;">{avg_stars:.2f} <span style="color:#FA4848; font-size:1.6rem;">★</span></div>
<div class="kpi-label">Average Rating</div>
</div>
<div class="kpi-card-white">
<div class="kpi-number" style="color: #0396BC;">{tot_revs:,}</div>
<div class="kpi-label">Total Customer Reviews</div>
</div>
<div class="kpi-card-white">
<div class="kpi-number" style="color: #029E6A;">{pos_pct:.1f}%</div>
<div class="kpi-label">Positive Sentiment ({pos_cnt:,})</div>
</div>
<div class="kpi-card-white">
<div class="kpi-number" style="color: #D71616;">{neg_pct:.1f}%</div>
<div class="kpi-label">Critical Complaints ({neg_cnt:,})</div>
</div>
</div>"""
    st.markdown(kpi_html, unsafe_allow_html=True)

    # 66% / 33% YELP BUSINESS PAGE GRID LAYOUT
    col_main, col_sidebar = st.columns([0.65, 0.35], gap="large")

    # LEFT COLUMN (66%): REVIEWS & SENTIMENT ANALYTICS
    with col_main:
        tab_reviews_view, tab_archive = st.tabs([
            "📊 Sentiment & Review Analytics", 
            "📝 Feedback Log & Archive"
        ])

        with tab_reviews_view:
            st.markdown(f"### Customer Reviews Breakdown for **{selected_biz_name}** ({city_st})")
            st.caption("Ordered: Positive Reviews FIRST, followed by Critical Complaints SECOND.")
            st.markdown("<br>", unsafe_allow_html=True)

            # SECTION 1: POSITIVE PRAISE REVIEWS (FIRST)
            st.markdown("""<div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
<span style="font-size:1.3rem;">🌟</span>
<h3 style="margin:0; color:#029E6A; font-family:'Poppins',sans-serif;">Positive Praise Reviews (4 & 5 Stars)</h3>
</div>""", unsafe_allow_html=True)

            if pos_reviews.empty:
                st.info("No positive reviews recorded for this business location.")
            else:
                for idx, row in pos_reviews.iterrows():
                    st.markdown(f"""<div class="review-card-pos">
<div style="display:flex; justify-content:space-between; align-items:center;">
<div>
<span style="color:#FA4848; font-weight:bold;">{render_stars(row['stars'])}</span>
<span style="font-weight:700; color:#2D2E2F; font-size:0.95rem; margin-left:6px;">{row['stars']:.1f} Stars</span>
</div>
<span class="review-date">🗓️ {row['review_date']}</span>
</div>
<div class="review-text">{row['document_text']}</div>
</div>""", unsafe_allow_html=True)

            st.markdown("<br><hr style='border-color:#F0F0F0;'><br>", unsafe_allow_html=True)

            # SECTION 2: CRITICAL COMPLAINTS REVIEWS (SECOND)
            st.markdown("""<div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
<span style="font-size:1.3rem;">🚨</span>
<h3 style="margin:0; color:#D71616; font-family:'Poppins',sans-serif;">Critical Customer Complaints (1 & 2 Stars)</h3>
</div>""", unsafe_allow_html=True)

            if neg_reviews.empty:
                st.success("🎉 No critical negative complaints recorded!")
            else:
                for idx, row in neg_reviews.iterrows():
                    st.markdown(f"""<div class="review-card-neg">
<div style="display:flex; justify-content:space-between; align-items:center;">
<div>
<span style="color:#FA4848; font-weight:bold;">{render_stars(row['stars'])}</span>
<span style="font-weight:700; color:#D71616; font-size:0.95rem; margin-left:6px;">{row['stars']:.1f} Stars</span>
</div>
<span class="review-date">🗓️ {row['review_date']}</span>
</div>
<div class="review-text">{row['document_text']}</div>
</div>""", unsafe_allow_html=True)

        with tab_archive:
            st.markdown(f"### Full SQL Feedback Log: {selected_biz_name} ({city_st})")
            if selected_biz_id:
                df_full_log = con.execute(f"""
                    SELECT review_id, stars, sentiment, review_date, document_text
                    FROM read_parquet('{gold_rev_glob}')
                    WHERE business_id = ?
                    ORDER BY review_date DESC
                    LIMIT 100
                """, [selected_biz_id]).df()
            else:
                df_full_log = con.execute(f"""
                    SELECT review_id, stars, sentiment, review_date, document_text
                    FROM read_parquet('{gold_rev_glob}')
                    WHERE business_name = ?
                    ORDER BY review_date DESC
                    LIMIT 100
                """, [selected_biz_name]).df()

            if not df_full_log.empty:
                st.dataframe(df_full_log, use_container_width=True)
            else:
                st.info("No logs available.")

    # RIGHT COLUMN (33% STICKY SIDEBAR): AI OWNER COPILOT CHATBOT
    with col_sidebar:
        st.markdown(f"""<div class="sidebar-widget">
<h3 style="margin:0; font-family:'Poppins',sans-serif; color:#2D2E2F; font-size:1.2rem; display:flex; align-items:center; gap:8px;">
<span style="color:#FA4848;">🤖</span> AI Business Copilot
</h3>
<p style="font-size:0.85rem; color:#6B6D6F; margin-top:4px; margin-bottom:12px;">
Tuned on {tot_revs:,} reviews for <strong>{selected_biz_name} ({city_st})</strong>
</p>
</div>""", unsafe_allow_html=True)

        copilot_key = f"owner_copilot_{selected_biz_id if selected_biz_id else selected_biz_name}"
        if copilot_key not in st.session_state:
            st.session_state[copilot_key] = [
                {
                    "role": "assistant",
                    "content": f"👋 Hi! I am your AI Copilot for **{selected_biz_name}** ({city_st}). Ask me anything about customer feedback, top complaints, praise, or staff service."
                }
            ]

        for msg in st.session_state[copilot_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)

        owner_prompt = st.chat_input(f"Ask AI Copilot about {selected_biz_name}...")

        if owner_prompt:
            st.session_state[copilot_key].append({"role": "user", "content": owner_prompt})
            with st.chat_message("user"):
                st.markdown(owner_prompt)

            with st.spinner(f"🤖 Retrieving reviews & generating AI insight for {selected_biz_name}..."):
                search_prompt = f"{selected_biz_name} {owner_prompt}" if selected_biz_name else owner_prompt
                rag_output = answer_question(
                    question=search_prompt,
                    doc_type="review",
                    business_id=selected_biz_id,   # ← CRITICAL: scope to this business
                    top_k=6,
                )
                ai_answer = rag_output["answer"]
                sources = rag_output["sources"]

                # Also pull a few DuckDB reviews for display
                if selected_biz_id:
                    biz_reviews_df = con.execute(f"""
                        SELECT stars, sentiment, review_date, document_text
                        FROM read_parquet('{gold_rev_glob}')
                        WHERE business_id = ?
                        ORDER BY review_date DESC
                        LIMIT 3
                    """, [selected_biz_id]).df()
                else:
                    biz_reviews_df = con.execute(f"""
                        SELECT stars, sentiment, review_date, document_text
                        FROM read_parquet('{gold_rev_glob}')
                        WHERE business_name = ?
                        ORDER BY review_date DESC
                        LIMIT 3
                    """, [selected_biz_name]).df()

            with st.chat_message("assistant"):
                # ── AI GENERATED INSIGHT ────────────────────────────────────
                copilot_reply = f"""### 🤖 AI Business Insight

{ai_answer}

---
### 🔍 Retrieved Review Evidence
"""
                if sources:
                    for i, src in enumerate(sources[:4], 1):
                        stars_val = src.get("stars") or src.get("business_rating", 0)
                        stars_val = float(stars_val) if stars_val else 0.0
                        s_label = str(src.get("sentiment", "")).upper()
                        date_val = src.get("review_date", src.get("date", "N/A"))
                        txt = str(src.get("document_text", "")).replace("\n", " ")
                        copilot_reply += f"""
* **Review #{i}** ({render_stars(stars_val)} `{stars_val:.1f}⭐` | `{s_label}` | `{date_val}`):
  > *"{txt[:240]}..."*
"""
                else:
                    copilot_reply += "\n*No relevant reviews were retrieved for this query.*"

                st.markdown(copilot_reply)
                st.session_state[copilot_key].append({"role": "assistant", "content": copilot_reply})

