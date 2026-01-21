import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import openai
from sklearn.ensemble import RandomForestRegressor
import io
import re
import unicodedata
import chardet
import streamlit.components.v1 as components
import base64

# --- 1. CONFIG & AUTHENTICATION (Must be first) ---
st.set_page_config(page_title="Enterprise AI Analytics", layout="wide")


def normalize(c: str) -> str:
    """
    Normalize column names for robust heuristic matching:
    - remove accents (Greek-safe)
    - lowercase
    - remove all non-alphanumeric characters except underscore
    """
    c = unicodedata.normalize("NFKD", c)
    c = "".join(ch for ch in c if not unicodedata.combining(ch))
    c = c.lower()
    c = re.sub(r"[^a-z0-9_]", "", c)  # keep underscore
    return c


def check_password():
    def password_entered():
        if st.session_state["password"] == "admin123":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "🔒Enter Admin Password",
            type="password",
            on_change=password_entered,
            key="password",
        )
        st.info("Hint: the password is 'admin123'")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "🔒Enter Admin Password",
            type="password",
            on_change=password_entered,
            key="password",
        )
        st.error("😕Wrong password")
        return False
    else:
        return True


if not check_password():
    st.stop()


@st.cache_data(show_spinner=False)
def load_data_file(file_bytes, filename):
    file = io.BytesIO(file_bytes)
    # --- Load file ---
    if filename.lower().endswith(".csv"):
        rawdata = file.read()
        file.seek(0)
        result = chardet.detect(rawdata)
        enc = result['encoding'] if result['encoding'] else 'utf-8-sig'
        # Αν είναι CSV με πιθανό ; separator
        try:
            df = pd.read_csv(io.BytesIO(rawdata), sep=',', encoding=enc)
            if len(df.columns) == 1:
                # Maybe semicolon separator
                df = pd.read_csv(io.BytesIO(rawdata), sep=';', encoding=enc)
        except Exception as e:
            st.error(f"CSV load error: {e}")
            return pd.DataFrame()


    else:
        engine = "openpyxl" if filename.lower().endswith(".xlsx") else None
        df = pd.read_excel(file, engine=engine)

    # --- Clean column names robustly ---
    df.columns = [
        re.sub(r"[^a-z0-9_]", "", unicodedata.normalize("NFKD", str(c).strip().replace("\xa0", "").lower()))
        for c in df.columns
    ]

    # --- Force rename required columns if any variant exists ---
    rename_map = {}
    for col in df.columns:
        c_norm = normalize(col)
        if any(x in c_norm for x in ["units", "sold", "posotita", "temaxia", "qty"]):
            rename_map[col] = "Units_Sold"
        elif any(x in c_norm for x in ["date", "imerominia", "hmeromhnia"]):
            rename_map[col] = "Date"
        elif any(x in c_norm for x in ["price", "timi", "lianiki"]):
            rename_map[col] = "Unit_Price"
        elif any(x in c_norm for x in ["product", "eidos", "proion", "sku"]):
            rename_map[col] = "Product"
        elif any(x in c_norm for x in ["cost", "kostos", "agora"]):
            rename_map[col] = "Cost_Per_Unit"

    df = df.rename(columns=rename_map)

    required = ["Date", "Units_Sold"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        return pd.DataFrame()

    # --- Convert types ---
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df["Units_Sold"] = pd.to_numeric(df["Units_Sold"], errors="coerce").fillna(0)
    if "Unit_Price" not in df.columns:
        df["Unit_Price"] = 0
    else:
        df["Unit_Price"] = pd.to_numeric(df["Unit_Price"], errors="coerce").fillna(0)
    if "Product" not in df.columns:
        df["Product"] = "General Item"

    df["Revenue"] = df["Units_Sold"] * df["Unit_Price"]
    df["Cost_Per_Unit"] = df.get("Cost_Per_Unit", df["Unit_Price"] * 0.7)
    df["Profit"] = df["Revenue"] - (df["Units_Sold"] * df["Cost_Per_Unit"])

    return df.dropna(subset=["Date"])


st.markdown(
    """
    <style>
    /* ---------- App Background ---------- */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e5e7eb;
        font-size: 14px;
    }

    /* ---------- Metric Cards (Scaled Down) ---------- */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.65) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(59, 130, 246, 0.22) !important;
        padding: 14px 16px !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.28) !important;
        transition: all 0.25s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.45) !important;
    }

    /* Metric typography */
    div[data-testid="stMetric"] label {
        font-size: 12px !important;
        color: #94a3b8 !important;
    }

    div[data-testid="stMetric"] div {
        font-size: 22px !important;
        font-weight: 600 !important;
        color: #f8fafc !important;
    }

    /* ---------- Tabs (More Compact) ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding-bottom: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 40px !important;
        background-color: rgba(51, 65, 85, 0.45) !important;
        border-radius: 10px !important;
        color: #9ca3af !important;
        padding: 0 16px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 3px 10px rgba(37, 99, 235, 0.35) !important;
    }

     /* Στυλ για το Orb */
    #floating-orb-id {
        position: fixed !important;
        bottom: 30px !important;
        right: 30px !important;
        width: 70px !important;
        height: 70px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #3b82f6, #2dd4bf) !important;
        color: white !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 30px !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.5) !important;
        z-index: 1000001 !important;
        cursor: pointer !important;
        border: 2px solid white !important;
        transition: transform 0.2s !important;
    }
    #floating-orb-id:hover { transform: scale(1.1); }

    /* Στυλ για το Μικρό Παράθυρο (Floating Window) */
    .floating-window-ui {
        position: fixed !important;
        bottom: 110px !important;
        right: 30px !important;
        width: 380px !important;
        height: 600px !important;
        background: #0f172a !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 20px !important;
        box-shadow: 0 20px 60px rgba(0,0,0,0.8) !important;
        z-index: 1000000 !important;
        display: flex !important;
        flex-direction: column !important;
        padding: 15px !important;
        overflow: hidden !important;
    }

    /* Εξαφάνιση των κενών του Streamlit */
    div[data-testid="stVerticalBlock"] > div:has(.floating-window),
    div[data-testid="stVerticalBlock"] > div:has(.floating-orb) {
        height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
    }

    /* Απόκρυψη των containers του Streamlit που περιέχουν το chat */
    div[data-testid="stVerticalBlock"] > div:has(.floating-chat-window),
    div[data-testid="stVerticalBlock"] > div:has(.orb-trigger) {
        height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
    }

     /* Εξαφάνιση του κενού που αφήνει το Streamlit */
     [data-testid="stVerticalBlock"] > div:has(.floating-chat-window) {
       height: 0 !important;
      margin: 0 !important;
      padding: 0 !important;

    }
    /* Custom Styles για τα στοιχεία μέσα στο παράθυρο */
    .chat-header {
        padding: 15px;
        background: linear-gradient(90deg, #1e293b, #0f172a);
        border-bottom: 1px solid #334155;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: white;
        font-weight: bold;
    }

    /* Μικραίνουμε το file uploader για να χωράει */
    div[data-testid="stFileUploader"] section {
        padding: 10px;
        background-color: rgba(255,255,255,0.05);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background-color: #0b1220 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.04);
    }

    /* ---------- Buttons (Less Dominant) ---------- */
    .stButton>button {
        border-radius: 8px !important;
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        padding: 6px 18px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        border: none !important;
        box-shadow: 0 3px 8px rgba(37, 99, 235, 0.25) !important;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        opacity: 0.92;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.35) !important;
    }

    /* ---------- Dataframes ---------- */
    .stDataFrame {
        font-size: 13px !important;
    }
    </style>
    <div id="voice-orb" onclick="toggleVoice()">🎙️</div>
    """, unsafe_allow_html=True)

# JavaScript για το Voice (Επικοινωνία με τον Browser)
components.html("""
<script>
    window.parent.startGlobalVoice = function() {
        const recognition = new (window.parent.SpeechRecognition || window.parent.webkitSpeechRecognition)();
        recognition.lang = 'el-GR';
        recognition.onresult = (event) => {
            const text = event.results[0][0].transcript;
            const chatInput = window.parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
            if (chatInput) {
                chatInput.value = text;
                chatInput.dispatchEvent(new Event('input', { bubbles: true }));
                setTimeout(() => {
                    const btn = window.parent.document.querySelector('button[data-testid="stChatInputButton"]');
                    if (btn) btn.click();
                }, 500);
            }
        };
        recognition.start();
    };
</script>
""", height=0)


def predict_sales_advanced(df):
    """
    Hyper-Accurate Self-Learning Algorithm:
    - Seasonal Decomposition (Day of week patterns)
    - Momentum Analysis (Trend detection)
    - Weighted Rolling Features (Focus on recent accuracy)
    """
    if "Units_Sold" not in df.columns or "Date" not in df.columns:
        return None

    df = df.copy().sort_values("Date")

    # 1. Feature Engineering: Seasonal & Temporal
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Is_Weekend'] = df['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)

    # 2. Advanced Learning: Momentum & EWMA
    # Ο EWMA δίνει μεγαλύτερη βαρύτητα στις πολύ πρόσφατες ημέρες (πιο "ζωντανό" learning)
    df['EWMA_3'] = df['Units_Sold'].ewm(span=3).mean()
    df['EWMA_7'] = df['Units_Sold'].ewm(span=7).mean()

    # Momentum: Είναι η διαφορά του σήμερα με το χθες (δείχνει την τάση)
    df['Momentum'] = df['Units_Sold'].diff()

    # Lag Features
    df['Lag_1'] = df['Units_Sold'].shift(1)
    df['Lag_7'] = df['Units_Sold'].shift(7)  # Weekly seasonality

    df = df.dropna()

    if len(df) < 14:  # Χρειαζόμαστε τουλάχιστον 2 εβδομάδες για ακρίβεια
        return None

    # 3. Training με βελτιστοποιημένες παραμέτρους
    features = [
        'Day_of_Week', 'Month', 'Is_Weekend',
        'EWMA_3', 'EWMA_7', 'Momentum', 'Lag_1', 'Lag_7'
    ]
    X = df[features]
    y = df['Units_Sold']

    # Χρησιμοποιούμε περισσότερους εκτιμητές για μείωση του σφάλματος
    model = RandomForestRegressor(
        n_estimators=1000,
        max_features='sqrt',
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X, y)

    # 4. Dynamic Prediction for Tomorrow
    last_row = df.iloc[-1]
    next_date = last_row['Date'] + pd.Timedelta(days=1)

    next_features = pd.DataFrame([{
        'Day_of_Week': next_date.dayofweek,
        'Month': next_date.month,
        'Is_Weekend': 1 if next_date.dayofweek >= 5 else 0,
        'EWMA_3': df['Units_Sold'].tail(3).ewm(span=3).mean().iloc[-1],
        'EWMA_7': df['Units_Sold'].tail(7).ewm(span=7).mean().iloc[-1],
        'Momentum': last_row['Units_Sold'] - df.iloc[-2]['Units_Sold'],
        'Lag_1': last_row['Units_Sold'],
        'Lag_7': df.iloc[-7]['Units_Sold']
    }])

    prediction = model.predict(next_features)[0]
    return max(0, float(prediction))


def ask_ai(df, query, key):
    """Call OpenAI using the modern Client interface (v1.0.0+)"""
    if not key:
        return "No OpenAI key provided."
    try:
        client = openai.OpenAI(api_key=key)

        if "Product" in df.columns and ("Units_Sold" in df.columns or "Profit" in df.columns):
            summ = df.groupby("Product")[["Units_Sold", "Profit"]].sum().fillna(0)
            summ_text = summ.to_string()
        else:
            summ_text = "No product-level data available."

        prompt = (
            f"Data summary:\n{summ_text}\n\n"
            f"Query: {query}\n"
            "Answer in Greek, concise."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"AI Error: {e}"


def run_mic_js():
    mic_code = """
    <script>
        var recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.lang = 'el-GR';
        recognition.start();

        recognition.onresult = function(event) {
            var transcript = event.results[0][0].transcript;

            // Στοχεύουμε το επίσημο Chat Input του Streamlit
            var chatInput = window.parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');

            if (chatInput) {
                chatInput.value = transcript;
                chatInput.dispatchEvent(new Event('input', { bubbles: true }));

                // Αυτόματο κλικ στο κουμπί αποστολής μετά από μισό δευτερόλεπτο
                setTimeout(() => {
                    var sendBtn = window.parent.document.querySelector('button[data-testid="stChatInputButton"]');
                    if(sendBtn) sendBtn.click();
                }, 500);
            }
        }
    </script>
    """
    components.html(mic_code, height=0)


def live_audio_logic():
    js_code = """
    <script>
    const orb = window.parent.document.getElementById('floating-orb');
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'el-GR';
    recognition.interimResults = false;

    function startRecognition() {
        orb.classList.add('recording');
        recognition.start();
    }

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        orb.classList.remove('recording');
        // Στέλνει το κείμενο στο Streamlit input
        const input = window.parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
        if (input) {
            input.value = transcript;
            input.dispatchEvent(new Event('input', { bubbles: true }));
            // Προαιρετικό: Αυτόματο πάτημα του Enter
            setTimeout(() => {
                const btn = window.parent.document.querySelector('button[data-testid="stChatInputButton"]');
                if (btn) btn.click();
            }, 500);
        }
    };

    recognition.onerror = () => orb.classList.remove('recording');
    recognition.onend = () => orb.classList.remove('recording');
    </script>
    """
    components.html(js_code, height=0)


live_audio_logic()


def inject_voice_js():
    js_code = """
    <script>
    function startVoice() {
        const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.lang = 'el-GR';

        recognition.onstart = () => {
            window.parent.postMessage({type: 'mic_status', status: 'listening'}, '*');
        };

        recognition.onresult = (event) => {
            const text = event.results[0][0].transcript;
            // Στοχεύουμε το textarea του Streamlit
            const textArea = window.parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
            if (textArea) {
                textArea.value = text;
                textArea.dispatchEvent(new Event('input', { bubbles: true }));
                // Αυτόματη αποστολή
                setTimeout(() => {
                    const btn = window.parent.document.querySelector('button[data-testid="stChatInputButton"]');
                    if (btn) btn.click();
                }, 600);
            }
        };
        recognition.start();
    }
    // Κάνουμε τη συνάρτηση διαθέσιμη στο parent window
    window.parent.startVoice = startVoice;
    </script>
    """
    components.html(js_code, height=0)


inject_voice_js()

# --- 4. SIDEBAR & NAVIGATION ---
with st.sidebar:
    st.title("💼Enterprise AI")
    st.caption("Advanced Business Intelligence")
    st.markdown("---")

    # Κύριο αρχείο δεδομένων (για τα γραφήματα)
    uploaded_file = st.file_uploader(
        "📂Upload Data Source (CSV, XLSX)", type=["csv", "xlsx", "ods"], key="main_data"
    )

    api_key = st.text_input("🔑OpenAI Key (optional)", type="password")

    st.markdown("---")

    # Κουμπί καθαρισμού
    if st.button("Clear Session"):
        keys_to_clear = ["messages", "price_change", "price_slider"]
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

# --- 5. MAIN APPLICATION ---
if uploaded_file:
    df = load_data_file(uploaded_file.getvalue(), uploaded_file.name)

    if df.empty:
        st.error("❌ Could not load the file. Please check column names.")
        st.stop()

    st.success("✅ File loaded successfully")
    st.write(df.head())

    t1, t2, = st.tabs(["Analytics", "Forecasts", ])

    with t1:
        st.subheader("📊Business Performance Dashboard")

        # KPIs
        rev = df["Revenue"].sum() if "Revenue" in df.columns else 0.0
        prof = df["Profit"].sum() if "Profit" in df.columns else 0.0
        margin = (prof / rev) * 100 if rev > 0 else 0.0

        if "Product" in df.columns and "Units_Sold" in df.columns:
            prod_group = df.groupby("Product")["Units_Sold"].sum()
            if len(prod_group) > 0:
                best_seller = prod_group.idxmax()
                units_top = int(prod_group.max())
            else:
                best_seller = "N/A"
                units_top = 0
        else:
            best_seller = "N/A"
            units_top = 0

        k1, k2, k3 = st.columns(3)
        k1.metric("Total Revenue", f"{rev:,.2f} €")
        k2.metric("Net Profit", f"{prof:,.2f} €", f"{margin:.1f}% Margin")
        k3.metric("Top Product", best_seller, f"{units_top} Units Sold")

        st.markdown("---")

        # Visuals
        col_chart1, col_chart2 = st.columns([2, 1])

        with col_chart1:
            st.subheader("📈 Revenue Analysis")
            if "Product" in df.columns and "Revenue" in df.columns:
                agg = df.groupby("Product")[["Revenue", "Profit"]].sum().reset_index()

                fig_bar = px.bar(
                    agg,
                    x="Product",
                    y="Revenue",
                    text_auto=".2s",
                    color="Profit",
                    color_continuous_scale=["#1e293b", "#3b82f6", "#2dd4bf"],
                    template="plotly_dark",
                )

                fig_bar.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False,
                    margin=dict(t=20, b=20, l=20, r=20),
                    xaxis=dict(showgrid=False, title=""),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Revenue (€)")
                )

                fig_bar.update_traces(
                    marker_line_width=0,
                    opacity=0.9
                )

                st.plotly_chart(fig_bar, width='stretch')
            else:
                st.info("Need 'Product' and 'Revenue' columns.")

        with col_chart2:
            st.subheader("🎯 Profit Share")
            if "Product" in df.columns and "Profit" in df.columns:
                fig_pie = px.pie(
                    df.groupby("Product")["Profit"].sum().reset_index(),
                    names="Product",
                    values="Profit",
                    hole=0.7,
                    template="plotly_dark",

                    color_discrete_sequence=["#3b82f6", "#2dd4bf", "#f43f5e", "#fbbf24", "#8b5cf6"]
                )

                fig_pie.update_traces(
                    textposition='outside',
                    textinfo='percent',
                    marker=dict(line=dict(color='#0f172a', width=2)),
                    pull=[0.02, 0.02, 0.02]
                )

                fig_pie.update_layout(
                    showlegend=False,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=30, b=30, l=30, r=30),
                    annotations=[dict(text='Profit', x=0.5, y=0.5, font_size=20, showarrow=False, font_color="#94a3b8")]
                )

                st.plotly_chart(fig_pie, width='stretch')
            else:
                st.info("Need 'Product' and 'Profit' columns.")

        st.subheader("📝Executive Summary (AI Generated)")
        if api_key:
            if st.button("🪄Generate Smart Insights"):
                with st.spinner("Analyzing business data..."):
                    insight_prompt = (
                        "Κάνε μια γρήγορη ανάλυση: 1. Ποιο προϊόν φέρνει το περισσότερο κέρδος; "
                        "2. Πού υπάρχει πρόβλημα; 3. Μια πρόταση για αύξηση πωλήσεων."
                    )
                    report = ask_ai(df, insight_prompt, api_key)
                    st.info(report)
        else:
            st.warning(
                "⚠️Provide an OpenAI API Key in the sidebar to unlock AI Insights."
            )

    # TAB 2: Advanced AI Forecasting
    with t2:
        st.subheader("🔮Strategic AI Forecast Comparison")

        if "price_change" not in st.session_state:
            st.session_state.price_change = 0

        st.session_state.price_change = st.slider(
            "Global Price Change to apply in forecasts (%)",
            -20,
            50,
            value=st.session_state.get("price_change", 0),
            key="price_slider",
        )
        st.info(
            f"Price change {st.session_state.price_change}% will affect forecasts in the Forecasts tab."
        )

        s1, s2 = st.columns(2)
        p_change = s1.slider(
            "Price Change (%) for Simulation", -20, 50, 0, key="sim_price"
        )
        c_change = s2.slider(
            "Cost Change (%) for Simulation", -20, 50, 0, key="sim_cost"
        )

        if (
                "Unit_Price" in df.columns
                and "Cost_Per_Unit" in df.columns
                and "Units_Sold" in df.columns
        ):
            current_profit = (
                    df["Units_Sold"] * (df["Unit_Price"] - df["Cost_Per_Unit"])
            ).sum()
            sim_profit = (
                    df["Units_Sold"]
                    * (
                            (df["Unit_Price"] * (1 + p_change / 100))
                            - (df["Cost_Per_Unit"] * (1 + c_change / 100))
                    )
            ).sum()
            diff = sim_profit - current_profit

            m1, m2, m3 = st.columns(3)
            m1.metric("Current Profit", f"{current_profit:,.2f} €")
            m2.metric(
                "Simulated Profit", f"{sim_profit:,.2f} €", delta=f"{diff:,.2f} €"
            )
            if diff > 0:
                m3.success("✅Profitable Scenario")
            elif diff < 0:
                m3.error("⚠️Loss Scenario")
            else:
                m3.info("No change")

            comp_data = pd.DataFrame(
                {
                    "Scenario": ["Current", "Simulated"],
                    "Profit": [current_profit, sim_profit],
                }
            )
            fig_sim = px.bar(
                comp_data,
                x="Scenario",
                y="Profit",
                color="Scenario",
                color_discrete_map={"Current": "#3b82f6", "Simulated": "#2dd4bf"},
                template="plotly_dark"
            )

            fig_sim.update_traces(
                marker_line_color='rgba(255,255,255,0.2)',
                marker_line_width=1.5,
                opacity=0.9,
                marker_pattern_shape=""
            )

            fig_sim.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20),
                xaxis=dict(
                    showgrid=False,
                    linecolor="rgba(255,255,255,0.1)",
                    title=""
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.05)",
                    linecolor="rgba(255,255,255,0.1)",
                    title="Profit (€)"
                ),
                hovermode="x"
            )

            st.plotly_chart(fig_sim, width='stretch')
        else:
            st.warning(
                "Simulator requires 'Unit_Price', 'Cost_Per_Unit' and 'Units_Sold' columns."
            )

        if "Units_Sold" in df.columns and "Date" in df.columns:
            df_ml = df.copy().sort_values("Date")
            df_ml = df_ml.reset_index(drop=True)
            df_ml["DayOfWeek"] = df_ml["Date"].dt.dayofweek
            df_ml["Month"] = df_ml["Date"].dt.month
            df_ml["IsWeekend"] = df_ml["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)
            df_ml["Day_Index"] = np.arange(len(df_ml))
            unit_profit = (df["Unit_Price"] - df["Cost_Per_Unit"]).mean()

            features = ["Day_Index", "DayOfWeek", "Month", "IsWeekend"]
            X = df_ml[features]
            y = df_ml["Units_Sold"]

            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X, y)

            last_date = df_ml["Date"].max()
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
            last_idx = int(df_ml["Day_Index"].max())

            future_df_input = pd.DataFrame(
                {
                    "Day_Index": list(range(last_idx + 1, last_idx + 8)),
                    "Date": future_dates,
                }
            )
            future_df_input["DayOfWeek"] = future_df_input["Date"].dt.dayofweek
            future_df_input["Month"] = future_df_input["Date"].dt.month
            future_df_input["IsWeekend"] = future_df_input["DayOfWeek"].apply(
                lambda x: 1 if x >= 5 else 0
            )

            base_preds = model.predict(future_df_input[features])

            p_change = st.session_state.get("price_change", 0)
            elasticity = -1.5
            impact = (p_change / 100) * elasticity
            strategic_preds = base_preds * (1 + impact)

            future_base = pd.DataFrame({
                "Date": future_dates,
                "Profit": base_preds * unit_profit,  # Πολλαπλασιασμός με unit_profit
                "Series": "Base Forecast (No Change)",
            })

            future_strategic = pd.DataFrame({
                "Date": future_dates,
                "Profit": strategic_preds * unit_profit,  # Πολλαπλασιασμός με unit_profit
                "Series": "Strategic Forecast (Price Adjusted)",
            })

            actual_line = pd.DataFrame({
                "Date": df_ml["Date"],
                "Profit": y * unit_profit,  # Πολλαπλασιασμός με unit_profit
                "Series": "Actual History"
            })

            last_real_point = actual_line.tail(1).copy()

            base_connected = pd.concat(
                [
                    last_real_point.assign(Series="Base Forecast (No Change)"),
                    future_base,
                ],
                ignore_index=True,
            )
            strat_connected = pd.concat(
                [
                    last_real_point.assign(
                        Series="Strategic Forecast (Price Adjusted)"
                    ),
                    future_strategic,
                ],
                ignore_index=True,
            )

            all_plots = pd.concat(
                [actual_line, base_connected, strat_connected], ignore_index=True
            )

            # Υπολογισμός του σκορ πριν το γράφημα
            model_accuracy = model.score(X, y) * 100

            fig = px.line(
                all_plots,
                x="Date",
                y="Profit",  # <--- ΕΔΩ ΗΤΑΝ ΤΟ ΛΑΘΟΣ, άλλαξέ το από "Units" σε "Profit"
                color="Series",
                line_shape="spline",
                render_mode="svg",
                color_discrete_sequence=["#3b82f6", "#f43f5e", "#2dd4bf"],
                template="plotly_dark"
            )

            badge_color = "#10b981" if model_accuracy > 80 else "#fbbf24"
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.95,
                text=f"🎯 Prediction Success Rate: {model_accuracy:.1f}%",
                showarrow=False,
                font=dict(size=14, color="#10b981"),
                bgcolor="rgba(16, 185, 129, 0.1)",
                bordercolor="#10b981",
                borderwidth=1,
                borderpad=7,
                standoff=10
            )

            fig.update_traces(
                line=dict(width=4),
                marker=dict(size=8, opacity=0.8, line=dict(width=1, color='white')),
                connectgaps=True
            )

            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",  # Διαφανές φόντο
                paper_bgcolor="rgba(0,0,0,0)",  # Διαφανές χαρτί
                font=dict(color="#94a3b8"),  # Χρώμα γραμματοσειράς Inter
                hovermode="x unified",  # Ενιαίο tooltip κατά το hover
                margin=dict(t=50, b=50, l=20, r=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor="rgba(0,0,0,0)"
                ),
                xaxis=dict(
                    showgrid=False,
                    linecolor="rgba(255,255,255,0.1)",
                    tickfont=dict(size=12)
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.05)",  # Πολύ απαλές γραμμές πλέγματος
                    linecolor="rgba(255,255,255,0.1)",
                    zeroline=False
                )
            )

            # Εμφάνιση
            st.plotly_chart(fig, width='stretch')
            diff_units = strategic_preds.sum() - base_preds.sum()
            col_res1, col_res2 = st.columns(2)
            col_res1.metric("Projected Weekly Units", f"{strategic_preds.sum():,.0f}")
            col_res2.metric(
                "Impact of Price Change",
                f"{diff_units:,.0f} Units",
                f"{impact * 100:.1f}%",
            )
        else:
            st.error(
                "⚠️This tab requires 'Date' and 'Units_Sold' columns in the dataset."
            )

else:
    st.title("👋Welcome to Enterprise AI")
    st.markdown(
        "Please verify identity to proceed and upload a dataset from the sidebar to unlock the dashboard."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

if "hidden_trigger" not in st.session_state:
    st.session_state.hidden_trigger = False

# --- 2. CSS for ACS-like floating chat window ---
st.markdown("""
<style>
/* hide the internal trigger block placeholder */
div[data-testid="stVerticalBlock"] > div:has(button[key="hidden_trigger"]) {
    display: none !important;
    position: absolute !important;
    height: 0 !important;
    width: 0 !important;
    overflow: hidden !important;
}

/* Floating container base - will be moved into parent by JS */
.floating-window-ui {
    position: fixed !important;
    bottom: 110px !important;
    right: 30px !important;
    width: 380px !important;
    height: 550px !important;
    z-index: 1000000 !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    box-shadow: 0 24px 48px rgba(7, 13, 26, 0.65) !important;
    font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

/* header (ACS red) */
.floating-window-ui .aci-header {
    background: linear-gradient(0deg,#c60b0b,#e31b1b) !important;
    padding: 12px 14px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}

/* header title */
.floating-window-ui .aci-header .title {
    color: white !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}

/* header close / icons */
.floating-window-ui .aci-header .icons {
    display:flex;
    gap:8px;
    align-items:center;
}
.floating-window-ui .aci-header .icon-btn {
    width:32px; height:32px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    background: rgba(255,255,255,0.12); color:white; cursor:pointer;
    border: none; font-size:14px;
}

/* body (white rounded content area) */
.floating-window-ui .aci-body {
    background: #ffffff !important;
    padding: 14px !important;
    flex: 1 1 auto !important;
    overflow-y: auto !important;
}

/* user and assistant bubbles */
.aci-msg { max-width: 84%; margin-bottom: 12px; display:block; clear: both; }
.aci-msg .bubble { padding: 10px 12px; border-radius: 12px; line-height:1.25; font-size:14px; box-shadow: 0 6px 18px rgba(15,23,42,0.06); }
.aci-msg.user { text-align: right; }
.aci-msg.user .bubble { background: #e5f2ff; color:#0f172a; margin-left:auto; border-top-right-radius:4px; }
.aci-msg.assistant { text-align:left; }
.aci-msg.assistant .bubble { background: #f8f9fb; color:#0f172a; margin-right:auto; border-top-left-radius:4px; display:inline-block; }

/* assistant mini header inside body: avatar + label */
.aci-assistant-label { display:flex; gap:10px; align-items:center; margin-bottom:8px; }
.aci-avatar { width:36px; height:36px; border-radius:50%; background:linear-gradient(135deg,#ff7a7a,#ff3b3b); display:flex; align-items:center; justify-content:center; color:white; font-weight:700; }

/* action buttons (rounded outlined) */
.aci-actions { display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }
.aci-action {
    border-radius: 999px; padding:8px 14px; border:2px solid #e31b1b; color:#e31b1b;
    background:transparent; font-weight:600; cursor:pointer; text-decoration:none;
}

/* input area */
.aci-input-wrap {
    display:flex; gap:8px; padding:12px; align-items:center; background:#0f172a; border-top:1px solid rgba(255,255,255,0.04);
}
.aci-input {
    flex:1; border-radius:999px; padding:10px 14px; border: none; outline:none;
    font-size:14px; background: rgba(255,255,255,0.06); color: white;
}
.aci-send-btn {
    width:46px; height:40px; border-radius: 999px; border:none; cursor:pointer;
    background: linear-gradient(135deg,#3b82f6,#2dd4bf); color:white; font-weight:700;
}

/* small responsive tweak */
@media (max-width:420px) {
    .floating-window-ui { width: 92% !important; right:4% !important; }
}
</style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown(
        "<div style='height:0; overflow:hidden; position:absolute;'>",
        unsafe_allow_html=True
    )
    st.button("INTERNAL_TRIGGER", key="hidden_trigger")
    st.markdown("</div>", unsafe_allow_html=True)

# --- 4. Orb script (same concept, show ❌ when open) ---
st.components.v1.html(f"""
<script>
(function() {{
    const parentDoc = window.parent.document;
    let orb = parentDoc.getElementById('permanent-orb');
    // Create orb if missing
    if (!orb) {{
        orb = parentDoc.createElement('div');
        orb.id = 'permanent-orb';
        Object.assign(orb.style, {{
            position: 'fixed', bottom: '30px', right: '30px',
            width: '70px', height: '70px', borderRadius: '50%',
            background: 'linear-gradient(135deg, #3b82f6, #2dd4bf)',
            color: 'white', fontSize: '28px', display: 'flex',
            alignItems: 'center', justifyContent: 'center',
            cursor: 'pointer', zIndex: '1000001',
            boxShadow: '0 10px 30px rgba(0,0,0,0.5)', border: '2px solid white'
        }});
        parentDoc.body.appendChild(orb);
        orb.onclick = function() {{
            const btns = parentDoc.querySelectorAll('button');
            const trigger = Array.from(btns).find(b => b.innerText.includes('INTERNAL_TRIGGER'));
            if (trigger) {{ trigger.click(); }}
        }};
    }}
    // reflect open/closed state (text only)
    orb.innerHTML = "{'❌' if st.session_state.chat_open else '🤖'}";
}})();
</script>
""", height=0)

# --- 5. Extra window content (ACS-style) ---
if st.session_state.chat_open:
    # build HTML chat window (messages rendered from st.session_state.messages)
    messages_html = ""
    for m in st.session_state.messages:
        role = "assistant" if m["role"] == "assistant" else "user"
        # escape content minimally for safety
        content = (m["content"]
                   .replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace("\n", "<br>"))
        messages_html += f'''
            <div class="aci-msg {role}">
                <div class="bubble">{content}</div>
            </div>
        '''

    # Compose the window HTML (header, body with messages + action buttons, input area)
    st.markdown(f"""
    <div id="chat-window-wrapper" class="floating-window-ui" >
        <div class="aci-header">
            <div style="display:flex; gap:10px; align-items:center;">
                <div style="width:36px;height:36px;border-radius:6px;background:rgba(255,255,255,0.08);display:flex;align-items:center;justify-content:center;color:white;font-weight:700;">AI</div>
                <div class="title">ACiStant<br/><span style="font-weight:400; font-size:11px; opacity:0.92;">Η ψηφιακή βοηθός σου</span></div>
            </div>
            <div class="icons">
                <div class="icon-btn" id="aci-minimize" title="Minimize">–</div>
                <div class="icon-btn" id="aci-close" title="Close">✕</div>
            </div>
        </div>

        <div class="aci-body" id="aci-body">
            <div class="aci-assistant-label">
                <div class="aci-avatar">A</div>
                <div style="font-weight:700;">Καλώς ήρθες στην ACS!</div>
            </div>

            <div class="aci-actions">
                <a class="aci-action" href="javascript:void(0)">My favourite ACS Locker</a>
                <a class="aci-action" href="javascript:void(0)">Αναζήτηση Αποστολής</a>
                <a class="aci-action" href="javascript:void(0)">Αναζήτηση Καταστήματος</a>
                <a class="aci-action" href="javascript:void(0)">Αλλαγή τρόπου παραλαβής / οδηγίες παράδοσης</a>
            </div>

            <div id="aci-messages" style="margin-top:12px;">
                {messages_html}
            </div>
        </div>

        <!-- input area rendered by Streamlit text_input below; keep separate so Python can receive input -->
        <div style="background:#0f172a;padding:8px;border-top:1px solid rgba(255,255,255,0.04);">
            <div style="color:white;font-size:12px; opacity:0.9; padding:6px 10px; border-radius:8px; background:rgba(255,255,255,0.02);">
                Χρησιμοποίησε το πεδίο εισαγωγής στο κάτω μέρος για να στείλεις μήνυμα.
            </div>
        </div>
    </div>

    <script>
    (function() {{
        // move the chat wrapper into parent document and wire the close/minimize buttons to the hidden trigger
        const parentDoc = window.parent.document;
        const blocks = parentDoc.querySelectorAll('div[data-testid="stVerticalBlock"]');
        const chatWin = Array.from(blocks).find(b => b.innerHTML.includes('chat-window-wrapper'));
        if (chatWin) {{
            // Add floating class if not already moved
            chatWin.classList.add('floating-window-ui');

            // append chatWin to body so it overlays everything
            try {{
                parentDoc.body.appendChild(chatWin);
            }} catch(e) {{
                // ignore cross-frame errors if any
            }}

            // find the INTERNAL_TRIGGER button in the parent and attach click handlers
            const trigger = Array.from(parentDoc.querySelectorAll('button')).find(b => b.innerText.includes('INTERNAL_TRIGGER'));

            // close button should toggle the trigger (so it behaves exactly like the orb)
            const closeBtn = chatWin.querySelector('#aci-close');
            if (closeBtn && trigger) {{
                closeBtn.onclick = function() {{ trigger.click(); }};
            }}

            // minimize just hides (does not change session state) - toggles display
            const minBtn = chatWin.querySelector('#aci-minimize');
            if (minBtn) {{
                minBtn.onclick = function() {{
                    const body = chatWin.querySelector('.aci-body');
                    if (!body) return;
                    if (body.style.display === 'none') {{
                        body.style.display = '';
                        chatWin.style.height = '550px';
                    }} else {{
                        body.style.display = 'none';
                        chatWin.style.height = '56px';
                    }}
                }};
            }}
        }}
    }})();
    </script>
    """, unsafe_allow_html=True)

    # Input field (Streamlit) under program control so Python can read it.
    user_input = st.text_input(
        "",
        placeholder="Ρωτήστε με...",
        key="aci_input",
        label_visibility="collapsed"
    )

    if user_input:
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": "Αναλύω τα δεδομένα σας..."}
        )
        st.session_state["aci_input"] = ""
        st.rerun()

# Logic for the hidden trigger toggling (unchanged)
if st.session_state.get('hidden_trigger'):
    st.session_state.chat_open = not st.session_state.chat_open
    st.rerun()

