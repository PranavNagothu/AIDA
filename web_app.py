import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import json
import os
import sys
from pathlib import Path
import time


# Add the current directory to path to import our agent
sys.path.append(str(Path(__file__).parent))

try:
    from data_analysis_agent import DataAnalysisAgent, DataAnalysisConfig
except ImportError:
    st.error("❌ Please ensure data_analysis_agent.py is in the same directory")
    st.info("Download both files and place them in the same folder")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AI Data Analysis Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/ai-data-analysis-agent',
        'Report a bug': "https://github.com/yourusername/ai-data-analysis-agent/issues",
        'About': "# AI Data Analysis Agent\nPowered by Llama 3 & LangGraph"
    }
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Main Header */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin: 2rem 0;
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Subtitle */
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        text-align: center;
        color: #64748b;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Feature Cards */
    .feature-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .feature-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .feature-description {
        color: #64748b;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Metric Cards */
    .metric-container {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 1.5 rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        flex: 1;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Insight and Recommendation Boxes */
    .insight-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 5px solid #3b82f6;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .insight-box:hover {
        transform: translateX(4px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 5px solid #22c55e;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .recommendation-box:hover {
        transform: translateX(4px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    /* Upload Area */
    .upload-area {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #3b82f6;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        border-radius: 10px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
    }
    
    .css-1d391kg .sidebar-content {
        color: white;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border-radius: 12px;
        border: 1px solid #cbd5e1;
        color: #475569;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: 1px solid #3b82f6;
    }
    
    /* Success/Warning/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 1px solid #22c55e;
        border-radius: 12px;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        border-radius: 12px;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 1px solid #ef4444;
        border-radius: 12px;
    }
    
    /* Animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Data Table Styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 0;
        color: #64748b;
        font-size: 0.9rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 4rem;
    }
    
    .footer a {
        color: #3b82f6;
        text-decoration: none;
        font-weight: 500;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
    
    /* Loading Animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        border: 4px solid #f3f4f6;
        border-top: 4px solid #3b82f6;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'groq_api_key' not in st.session_state:
        st.session_state.groq_api_key = ""
    if 'model_name' not in st.session_state:
        st.session_state.model_name = "llama3-70b-8192"
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

def create_agent():
    """Create and configure the data analysis agent"""
    try:
        # Check environment variable first, then session state
        groq_api_key = os.environ.get('GROQ_API_KEY') or st.session_state.get('groq_api_key', '')
        if not groq_api_key:
            return None
            
        agent = DataAnalysisAgent(
            groq_api_key=groq_api_key,
            model_name=st.session_state.get('model_name', 'llama3-70b-8192')
        )
        return agent
    except Exception as e:
        st.error(f"Failed to create agent: {str(e)}")
        return None

def sidebar_config():
    """Configure the beautiful sidebar"""
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <div style='font-size: 4.5rem; margin-bottom: 0 rem;'>🤖</div>
            <h1 style='
                background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #06b6d4 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin: 0; 
                font-size: 1.6rem;
                font-weight: 700;
            '>AI Agents on action</h1>
            <p style='color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Powered by Llama 3</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Check for environment variable first
        env_api_key = os.environ.get('GROQ_API_KEY')
        
        if env_api_key:
            st.success("✅ API Key Configured")
            st.session_state.groq_api_key = env_api_key
            api_key_configured = True
        else:
            st.subheader("🔑 API Setup")
            st.info("💡 Set GROQ_API_KEY environment variable")
            
            groq_api_key = st.text_input(
                "Groq API Key",
                type="password",
                value=st.session_state.groq_api_key,
                help="Get your API key from console.groq.com"
            )
            
            if groq_api_key:
                st.session_state.groq_api_key = groq_api_key
                api_key_configured = True
            else:
                api_key_configured = False
        
        st.markdown("---")
        
        # Model Selection
        st.subheader("🧠 AI Model")
        model_options = {
            "llama3-70b-8192": "Llama 3 70B (Recommended)",
            "llama3-8b-8192": "Llama 3 8B (Faster)",
            "mixtral-8x7b-32768": "Mixtral 8x7B"
        }
        
        selected_model = st.selectbox(
            "Choose Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0
        )
        st.session_state.model_name = selected_model
        
        st.markdown("---")
        
        # Analysis Options
        st.subheader("⚙️ Analysis Settings")
        
        industry_type = st.selectbox(
            "Industry Focus",
            ["General", "Retail", "Healthcare", "Finance", "Manufacturing", "Technology"],
            help="Customize insights for your industry"
        )
        st.session_state.industry_type = industry_type
        
        enable_advanced = st.toggle(
            "Advanced Analysis",
            value=True,
            help="Include correlation analysis and advanced insights"
        )
        st.session_state.enable_advanced = enable_advanced
        
        auto_insights = st.toggle(
            "Auto-Generate Insights",
            value=True,
            help="Automatically generate business insights"
        )
        st.session_state.auto_insights = auto_insights
        
        st.markdown("---")
        
        # Quick Stats with dynamic insights count
        if st.session_state.dataset is not None:
            st.subheader("📊 Dataset Info")
            df = st.session_state.dataset
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", f"{df.shape[0]:,}")
                st.metric("Columns", df.shape[1])
            with col2:
                st.metric("Missing", f"{df.isnull().sum().sum():,}")
                st.metric("Size", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Show insights count if analysis is complete (now shows top 5)
        if st.session_state.analysis_results:
            insights = st.session_state.analysis_results.get('insights', [])
            recommendations = st.session_state.analysis_results.get('recommendations', [])
            
            # Process to get clean counts (max 5 each)
            processed_insights_count = min(len([i for i in insights if isinstance(i, str) and len(i.strip()) > 20]), 5)
            processed_recommendations_count = min(len([r for r in recommendations if isinstance(r, str) and len(r.strip()) > 20]), 5)
            
            st.markdown("---")
            st.subheader("🧠 Analysis Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("💡 Top Insights", processed_insights_count)
            with col2:
                st.metric("🎯 Top Recommendations", processed_recommendations_count)
        
        st.markdown("---")
        
        # Help Section
        with st.expander("💡 Quick Help"):
            st.markdown("""
            **Supported Formats:**
            - CSV files (.csv)
            - Excel files (.xlsx, .xls)  
            - JSON files (.json)
            
            **Best Practices:**
            - Clean column names
            - Handle missing values
            - Include date columns
            - Mix numeric & categorical data
            
            **Need Help?**
            - [Documentation](https://github.com/yourusername/ai-data-analysis-agent)
            - [Examples](https://github.com/yourusername/ai-data-analysis-agent/examples)
            """)
    
    return api_key_configured

def display_hero_section():
    """Display the beautiful hero section"""
    st.markdown('<div class="main-header animate-fade-in">AIDA-AI Data Analyzer </div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="subtitle animate-fade-in">
        Transform your raw data into actionable business insights with the power of AI.<br>
        Upload, analyze, and discover patterns automatically using intelligent agents.
    </div>
    """, unsafe_allow_html=True)

def display_features():
    """Display feature cards"""
    st.markdown("### ✨ What This AI Agent Can Do")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Intelligent Analysis</div>
            <div class="feature-description">
                Our AI automatically understands your data structure, identifies patterns, 
                and generates meaningful insights without any manual configuration.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Smart Visualizations</div>
            <div class="feature-description">
                Intelligently creates the most appropriate charts and graphs for your data, with interactive visualizations.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Actionable Recommendations</div>
            <div class="feature-description">
                Get specific, measurable recommendations for improving your business 
                based on data-driven insights.
            </div>
        </div>
        """, unsafe_allow_html=True)

def upload_dataset():
    """Beautiful dataset upload section"""
    st.markdown("### 📊 Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Drag and drop your file here or click to browse",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        try:
            # Show loading spinner
            with st.spinner("🔍 Processing your dataset..."):
                time.sleep(1)  # Small delay for UX
                
                # Read the file based on extension
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                else:
                    st.error("Unsupported file format")
                    return False
            
            st.session_state.dataset = df
            st.session_state.uploaded_filename = uploaded_file.name
            
            # Success message
            st.success(f"✅ Successfully loaded **{uploaded_file.name}**")
            
            # Beautiful metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{df.shape[0]:,}</div>
                    <div class="metric-label">Rows</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{df.shape[1]}</div>
                    <div class="metric-label">Columns</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                missing = df.isnull().sum().sum()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{missing:,}</div>
                    <div class="metric-label">Missing Values</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                size_mb = df.memory_usage(deep=True).sum() / 1024**2
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{size_mb:.1f} MB</div>
                    <div class="metric-label">File Size</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Data preview with beautiful styling
            st.markdown("#### 📋 Data Preview")
            st.dataframe(
                df.head(10), 
                use_container_width=True,
                height=300
            )
            
            # Column information in expandable section
            with st.expander("📊 Detailed Column Information", expanded=False):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count(),
                    'Null Count': df.isnull().sum(),
                    'Unique Values': df.nunique(),
                    'Sample Data': [str(df[col].iloc[0]) if len(df) > 0 else '' for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True)
                
            return True
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            return False
    else:
        # Show upload placeholder
        st.markdown("""
        <div class="upload-area">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📁</div>
            <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">
                Drop your dataset here
            </div>
            <div style="color: #64748b;">
                Supports CSV, Excel, and JSON files • Max 200MB
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    return False

def run_analysis():
    """Run the AI analysis with beautiful progress indicators"""
    if st.session_state.dataset is None:
        st.warning("Please upload a dataset first.")
        return
    
    # Check for API key from environment or session state
    api_key = os.environ.get('GROQ_API_KEY') or st.session_state.get('groq_api_key')
    if not api_key:
        st.warning("Please set GROQ_API_KEY environment variable or enter it in the sidebar.")
        return
    
    # Create agent
    with st.spinner("🤖 Initializing AI agent..."):
        agent = create_agent()
        if agent is None:
            st.error("Failed to initialize AI agent. Check your API key.")
            return
    
    st.session_state.agent = agent
    
    # Save dataset temporarily
    temp_file = "temp_dataset.csv"
    st.session_state.dataset.to_csv(temp_file, index=False)
    
    # Beautiful progress tracking
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### 🚀 AI Analysis in Progress")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step indicators
        steps = [
            ("🔍", "Analyzing dataset structure"),
            ("📊", "Examining columns and data quality"),
            ("🧠", "Generating AI insights"),
            ("📈", "Planning visualizations"),
            ("🎨", "Creating charts"),
            ("🎯", "Formulating recommendations")
        ]
        
        step_cols = st.columns(len(steps))
        step_indicators = []
        
        for i, (icon, desc) in enumerate(steps):
            with step_cols[i]:
                step_indicators.append(st.empty())
                step_indicators[i].markdown(f"""
                <div style="text-align: center; padding: 1rem; opacity: 0.3;">
                    <div style="font-size: 2rem;">{icon}</div>
                    <div style="font-size: 0.8rem; margin-top: 0.5rem;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
    
    try:
        # Step 1
        step_indicators[0].markdown(f"""
        <div style="text-align: center; padding: 1rem; opacity: 1; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 12px;">
            <div style="font-size: 2rem;">🔍</div>
            <div style="font-size: 0.8rem; margin-top: 0.5rem; font-weight: 600;">Analyzing Structure</div>
        </div>
        """, unsafe_allow_html=True)
        status_text.markdown("**🔍 AI agent analyzing dataset structure...**")
        progress_bar.progress(15)
        time.sleep(1)
        
        # Step 2
        step_indicators[1].markdown(f"""
        <div style="text-align: center; padding: 1rem; opacity: 1; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 12px;">
            <div style="font-size: 2rem;">📊</div>
            <div style="font-size: 0.8rem; margin-top: 0.5rem; font-weight: 600;">Examining Data</div>
        </div>
        """, unsafe_allow_html=True)
        status_text.markdown("**📊 Analyzing columns and data quality...**")
        progress_bar.progress(30)
        time.sleep(1)
        
        # Step 3
        step_indicators[2].markdown(f"""
        <div style="text-align: center; padding: 1rem; opacity: 1; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 12px;">
            <div style="font-size: 2rem;">🧠</div>
            <div style="font-size: 0.8rem; margin-top: 0.5rem; font-weight: 600;">AI Thinking</div>
        </div>
        """, unsafe_allow_html=True)
        status_text.markdown("**🧠 Generating insights with AI...**")
        progress_bar.progress(50)
        time.sleep(1)
        
        # Step 4
        step_indicators[3].markdown(f"""
        <div style="text-align: center; padding: 1rem; opacity: 1; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 12px;">
            <div style="font-size: 2rem;">📈</div>
            <div style="font-size: 0.8rem; margin-top: 0.5rem; font-weight: 600;">Planning Charts</div>
        </div>
        """, unsafe_allow_html=True)
        status_text.markdown("**📈 Planning optimal visualizations...**")
        progress_bar.progress(70)
        time.sleep(1)
        
        # Step 5
        step_indicators[4].markdown(f"""
        <div style="text-align: center; padding: 1rem; opacity: 1; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 12px;">
            <div style="font-size: 2rem;">🎨</div>
            <div style="font-size: 0.8rem; margin-top: 0.5rem; font-weight: 600;">Creating Charts</div>
        </div>
        """, unsafe_allow_html=True)
        status_text.markdown("**🎨 Creating beautiful visualizations...**")
        progress_bar.progress(85)
        
        # Run the actual analysis
        results = agent.analyze_dataset(temp_file)
        
        # Step 6
        step_indicators[5].markdown(f"""
        <div style="text-align: center; padding: 1rem; opacity: 1; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 12px;">
            <div style="font-size: 2rem;">🎯</div>
            <div style="font-size: 0.8rem; margin-top: 0.5rem; font-weight: 600;">Final Recommendations</div>
        </div>
        """, unsafe_allow_html=True)
        status_text.markdown("**🎯 Formulating actionable recommendations...**")
        progress_bar.progress(100)
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        if "error" in results:
            st.error(f"❌ Analysis failed: {results['error']}")
            return
        
        st.session_state.analysis_results = results
        st.session_state.analysis_complete = True
        
        # Success animation
        status_text.markdown("**✅ Analysis completed successfully!**")
        
        # Show completion message
        st.balloons()
        time.sleep(1)
        
        # Clear progress and show results
        progress_container.empty()
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        if os.path.exists(temp_file):
            os.remove(temp_file)

def display_results():
    """Display beautiful analysis results"""
    results = st.session_state.analysis_results
    if results is None:
        return
    
    # Results header
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0;">
        <h1 style="font-size: 2.5rem; color: #1e293b; margin-bottom: 0.5rem;">📊 Analysis Complete!</h1>
        <p style="font-size: 1.1rem; color: #64748b;">Here are your AI-generated insights and recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset Overview with beautiful cards
    st.markdown("### 📋 Dataset Overview")
    info = results.get('dataset_info', {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = [
        ("📊", "Total Rows", f"{info.get('shape', [0])[0]:,}", "#3b82f6"),
        ("📋", "Columns", str(info.get('shape', [0, 0])[1]), "#8b5cf6"),
        ("🔢", "Numeric", str(len(info.get('numeric_columns', []))), "#06b6d4"),
        ("📝", "Categorical", str(len(info.get('categorical_columns', []))), "#10b981"),
        ("✨", "Quality Score", f"{max(0, 100 - (sum(info.get('null_counts', {}).values()) / max(info.get('shape', [1, 1])[0] * info.get('shape', [1, 1])[1], 1) * 100)):.0f}%", "#f59e0b")
    ]
    
    for i, (icon, label, value, color) in enumerate(metrics):
        with [col1, col2, col3, col4, col5][i]:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {color}15 0%, {color}25 100%);
                border: 2px solid {color}30;
                border-radius: 16px;
                padding: 1.5rem;
                text-align: center;
                margin: 0.5rem 0;
                transition: transform 0.2s ease;
            ">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: {color}; margin-bottom: 0.25rem;">{value}</div>
                <div style="font-size: 0.9rem; color: #64748b; font-weight: 500;">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Insights Section - Extract complete insights with headers and content combined
    st.markdown("### 💡 Key Insights")
    insights = results.get('insights', [])
    
    if insights:
        # Combine all insight text and parse properly
        full_text = ' '.join(str(item) for item in insights)
        
        # Extract complete insights (header + content) using regex
        import re
        
        # Pattern to match **Insight X:** followed by content until next insight or end
        insight_pattern = r'\*\*Insight (\d+):(.*?)(?=\*\*Insight \d+:|$)'
        matches = re.findall(insight_pattern, full_text, re.DOTALL)
        
        processed_insights = []
        for match in matches:
            insight_num, content = match
            clean_content = content.strip().rstrip('*')
            if len(clean_content) > 20:
                processed_insights.append(clean_content)
        
        # Take top 5 insights
        top_insights = processed_insights[:5]
        
        if top_insights:
            st.markdown(f"**Top {len(top_insights)} key insights from your data:**")
            st.markdown("<br>", unsafe_allow_html=True)
            
            for i, insight in enumerate(top_insights):
                st.markdown(f"""
                <div class="insight-box animate-fade-in">
                    <div style="display: flex; align-items: flex-start; gap: 1rem;">
                        <div style="
                            background: #3b82f6; 
                            color: white; 
                            border-radius: 50%; 
                            width: 32px; 
                            height: 32px; 
                            display: flex; 
                            align-items: center; 
                            justify-content: center; 
                            font-weight: bold; 
                            font-size: 0.9rem;
                            flex-shrink: 0;
                        ">{i+1}</div>
                        <div style="flex: 1;">
                            <strong style="color: #1e293b;">💡 Key Insight {i+1}:</strong><br>
                            <span style="color: #475569; line-height: 1.6;">{insight}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("🔍 No insights could be extracted from the analysis.")
    else:
        st.info("🔍 No insights were generated.")
    
    # Interactive Visualizations Section
    st.markdown("### 📈 Interactive Data Exploration")
    
    if st.session_state.dataset is not None:
        df = st.session_state.dataset
        
        # Beautiful tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Distributions", 
            "🔗 Correlations", 
            "📈 Trends & Patterns", 
            "🎯 Custom Analysis"
        ])
        
        with tab1:
            st.markdown("#### 📊 Distribution Analysis")
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) > 0:
                # Column selector at the top
                selected_col = st.selectbox(
                    "Select column to analyze",
                    numeric_cols,
                    key="dist_col"
                )
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Show all three plots side by side
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Histogram**")
                    fig_hist = px.histogram(
                        df, 
                        x=selected_col, 
                        title=f"Histogram",
                        nbins=30,
                        color_discrete_sequence=['#3b82f6']
                    )
                    fig_hist.update_layout(
                        height=380,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        title_font_size=14,
                        margin=dict(t=40, b=40, l=40, r=40)
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    st.markdown("**Box Plot**")
                    fig_box = px.box(
                        df, 
                        y=selected_col, 
                        title=f"Box Plot",
                        color_discrete_sequence=['#8b5cf6']
                    )
                    fig_box.update_layout(
                        height=380,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        title_font_size=14,
                        margin=dict(t=40, b=40, l=40, r=40)
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                
                with col3:
                    st.markdown("**Violin Plot**")
                    fig_violin = px.violin(
                        df, 
                        y=selected_col, 
                        title=f"Violin Plot",
                        color_discrete_sequence=['#06b6d4']
                    )
                    fig_violin.update_layout(
                        height=380,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        title_font_size=14,
                        margin=dict(t=40, b=40, l=40, r=40)
                    )
                    st.plotly_chart(fig_violin, use_container_width=True)
                
                # Statistics cards below the plots
                st.markdown("#### 📊 Statistical Summary")
                stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)
                
                stats = [
                    ("Mean", f"{df[selected_col].mean():.2f}", "#3b82f6"),
                    ("Median", f"{df[selected_col].median():.2f}", "#8b5cf6"),
                    ("Std Dev", f"{df[selected_col].std():.2f}", "#06b6d4"),
                    ("Min", f"{df[selected_col].min():.2f}", "#10b981"),
                    ("Max", f"{df[selected_col].max():.2f}", "#f59e0b")
                ]
                
                for i, (label, value, color) in enumerate(stats):
                    with [stats_col1, stats_col2, stats_col3, stats_col4, stats_col5][i]:
                        st.markdown(f"""
                        <div style="
                            background: {color}15;
                            border: 1px solid {color}30;
                            border-radius: 12px;
                            padding: 1rem;
                            text-align: center;
                        ">
                            <div style="font-size: 1.4rem; font-weight: 700; color: {color};">{value}</div>
                            <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.25rem;">{label}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("📊 No numeric columns found for distribution analysis.")
        
        with tab2:
            st.markdown("#### 🔗 Correlation Analysis")
            
            if len(numeric_cols) > 1:
                # Correlation matrix heatmap
                corr_matrix = df[numeric_cols].corr()
                
                fig = px.imshow(
                    corr_matrix, 
                    text_auto=True, 
                    aspect="auto",
                    title="Correlation Matrix",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1
                )
                fig.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Top correlations
                st.markdown("#### 🔗 Strongest Correlations")
                correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if not pd.isna(corr_val):
                            correlations.append({
                                'Variable 1': corr_matrix.columns[i],
                                'Variable 2': corr_matrix.columns[j],
                                'Correlation': corr_val,
                                'Strength': abs(corr_val)
                            })
                
                if correlations:
                    corr_df = pd.DataFrame(correlations)
                    corr_df = corr_df.sort_values('Strength', ascending=False).head(10)
                    
                    # Display as beautiful cards
                    for _, row in corr_df.head(5).iterrows():
                        strength = "Strong" if row['Strength'] > 0.7 else "Moderate" if row['Strength'] > 0.5 else "Weak"
                        color = "#ef4444" if row['Strength'] > 0.7 else "#f59e0b" if row['Strength'] > 0.5 else "#10b981"
                        
                        st.markdown(f"""
                        <div style="
                            background: {color}15;
                            border-left: 4px solid {color};
                            border-radius: 8px;
                            padding: 1rem;
                            margin: 0.5rem 0;
                        ">
                            <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">
                                {row['Variable 1']} ↔ {row['Variable 2']}
                            </div>
                            <div style="color: #64748b;">
                                Correlation: <strong style="color: {color};">{row['Correlation']:.3f}</strong> 
                                ({strength} relationship)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("🔗 Need at least 2 numeric columns for correlation analysis.")
        
        with tab3:
            st.markdown("#### 📈 Trends & Patterns")
            
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(date_cols) > 0 and len(numeric_cols) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    date_col = st.selectbox("Date column", date_cols, key="trend_date")
                with col2:
                    value_col = st.selectbox("Value column", numeric_cols, key="trend_value")
                
                df_sorted = df.sort_values(date_col)
                fig = px.line(
                    df_sorted, 
                    x=date_col, 
                    y=value_col, 
                    title=f"{value_col} Over Time",
                    color_discrete_sequence=['#3b82f6']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            elif cat_cols and numeric_cols:
                st.markdown("#### 📊 Category-based Analysis")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    cat_col = st.selectbox("Category", cat_cols, key="cat_trend")
                with col2:
                    num_col = st.selectbox("Numeric value", numeric_cols, key="num_trend")
                with col3:
                    agg_func = st.selectbox("Aggregation", ["mean", "sum", "count", "median"])
                
                if agg_func == "count":
                    grouped = df.groupby(cat_col).size().reset_index(name='count')
                    y_col = 'count'
                else:
                    grouped = df.groupby(cat_col)[num_col].agg(agg_func).reset_index()
                    y_col = num_col
                
                fig = px.bar(
                    grouped, 
                    x=cat_col, 
                    y=y_col,
                    title=f"{agg_func.title()} of {num_col if agg_func != 'count' else 'Count'} by {cat_col}",
                    color_discrete_sequence=['#8b5cf6']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📈 Upload data with date columns or categorical data to see trends.")
        
        with tab4:
            st.markdown("#### 🎯 Custom Analysis Builder")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                viz_type = st.selectbox(
                    "Chart Type",
                    ["Scatter Plot", "Bar Chart", "Pie Chart", "Sunburst", "Treemap"]
                )
                
                if viz_type == "Scatter Plot" and len(numeric_cols) >= 2:
                    x_col = st.selectbox("X-axis", numeric_cols, key="custom_x")
                    y_col = st.selectbox("Y-axis", numeric_cols, key="custom_y")
                    color_col = st.selectbox("Color by", ["None"] + list(df.columns), key="custom_color")
                    size_col = st.selectbox("Size by", ["None"] + numeric_cols, key="custom_size")
                
                elif viz_type in ["Bar Chart", "Pie Chart"] and cat_cols:
                    cat_col = st.selectbox("Category", cat_cols, key="custom_cat")
                    if numeric_cols:
                        val_col = st.selectbox("Value (optional)", ["Count"] + numeric_cols, key="custom_val")
                    else:
                        val_col = "Count"
            
            with col2:
                try:
                    if viz_type == "Scatter Plot" and len(numeric_cols) >= 2:
                        fig = px.scatter(
                            df,
                            x=x_col,
                            y=y_col,
                            color=None if color_col == "None" else color_col,
                            size=None if size_col == "None" else size_col,
                            title=f"{y_col} vs {x_col}",
                            color_discrete_sequence=['#3b82f6'],
                            hover_data=df.columns[:5].tolist()
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Pie Chart" and cat_cols:
                        if val_col == "Count":
                            value_counts = df[cat_col].value_counts().head(8)
                            fig = px.pie(
                                values=value_counts.values,
                                names=value_counts.index,
                                title=f"Distribution of {cat_col}"
                            )
                        else:
                            grouped = df.groupby(cat_col)[val_col].sum().head(8)
                            fig = px.pie(
                                values=grouped.values,
                                names=grouped.index,
                                title=f"{val_col} by {cat_col}"
                            )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")
    
    # Recommendations Section - Extract complete recommendations with headers and content combined
    st.markdown("### 🎯 AI-Generated Recommendations")
    recommendations = results.get('recommendations', [])
    
    if recommendations:
        # Combine all recommendation text and parse properly
        full_text = ' '.join(str(item) for item in recommendations)
        
        # Extract complete recommendations using regex
        import re
        
        # Pattern to match recommendations (various formats)
        rec_patterns = [
            r'\*\*.*?(\d+):(.*?)(?=\*\*.*?\d+:|$)',  # **Something 1:** format
            r'(\d+)\.\s+(.*?)(?=\d+\.|$)',           # 1. format
        ]
        
        processed_recommendations = []
        for pattern in rec_patterns:
            matches = re.findall(pattern, full_text, re.DOTALL)
            if matches:
                for match in matches:
                    if len(match) == 2:
                        rec_num, content = match
                        clean_content = content.strip().rstrip('*')
                        if len(clean_content) > 20:
                            processed_recommendations.append(clean_content)
                break
        
        # Take top 5 recommendations
        top_recommendations = processed_recommendations[:5]
        
        if top_recommendations:
            st.markdown(f"**Top {len(top_recommendations)} actionable recommendations:**")
            st.markdown("<br>", unsafe_allow_html=True)
            
            for i, rec in enumerate(top_recommendations):
                st.markdown(f"""
                <div class="recommendation-box animate-fade-in">
                    <div style="display: flex; align-items: flex-start; gap: 1rem;">
                        <div style="
                            background: #22c55e; 
                            color: white; 
                            border-radius: 50%; 
                            width: 32px; 
                            height: 32px; 
                            display: flex; 
                            align-items: center; 
                            justify-content: center; 
                            font-weight: bold; 
                            font-size: 0.9rem;
                            flex-shrink: 0;
                        ">{i+1}</div>
                        <div style="flex: 1;">
                            <strong style="color: #1e293b;">🎯 Recommendation {i+1}:</strong><br>
                            <span style="color: #475569; line-height: 1.6;">{rec}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("🎯 No recommendations could be extracted from the analysis.")
    else:
        st.info("🎯 No recommendations were generated.")
    
    # Download Results Section
    st.markdown("### 💾 Download Your Results")
    
    col1, col2, col3 = st.columns(3)
    
    download_items = [
        ("📄", "Analysis Report (JSON)", "Download complete analysis", "json"),
        ("📊", "Enhanced Dataset (CSV)", "Download processed data", "csv"),
        ("📋", "Executive Summary (MD)", "Download business report", "md")
    ]
    
    for i, (icon, title, desc, file_type) in enumerate(download_items):
        with [col1, col2, col3][i]:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border: 2px solid #e2e8f0;
                border-radius: 16px;
                padding: 1.5rem;
                text-align: center;
                margin: 0.5rem 0;
                transition: all 0.3s ease;
            ">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">{icon}</div>
                <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; color: #1e293b;">{title}</div>
                <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 1rem;">{desc}</div>
            """, unsafe_allow_html=True)
            
            if file_type == "json":
                data = json.dumps(results, indent=2, default=str)
                filename = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                mime = "application/json"
            elif file_type == "csv":
                data = st.session_state.dataset.to_csv(index=False)
                filename = f"enhanced_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                mime = "text/csv"
            else:  # md
                data = generate_report(results)
                filename = f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                mime = "text/markdown"
            
            st.download_button(
                label=f"Download {file_type.upper()}",
                data=data,
                file_name=filename,
                mime=mime,
                use_container_width=True
            )
            
            st.markdown("</div>", unsafe_allow_html=True)

def generate_report(results):
    """Generate a beautiful markdown report"""
    filename = getattr(st.session_state, 'uploaded_filename', 'dataset')
    
    report = f"""# 🤖 AI Data Analysis Executive Summary

**Dataset:** {filename}  
**Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}  
**Powered by:** Llama 3 & LangGraph AI Agents

---

## 📊 Executive Overview

This report presents key findings from an AI-powered analysis of your dataset. Our advanced language models have identified patterns, trends, and opportunities that can drive business decisions.

### Dataset Metrics
- **Total Records:** {results.get('dataset_info', {}).get('shape', [0])[0]:,}
- **Data Points:** {len(results.get('dataset_info', {}).get('columns', []))}
- **Data Quality Score:** {max(0, 100 - (sum(results.get('dataset_info', {}).get('null_counts', {}).values()) / max(results.get('dataset_info', {}).get('shape', [1, 1])[0] * results.get('dataset_info', {}).get('shape', [1, 1])[1], 1) * 100)):.0f}%

---

## 💡 Strategic Insights

Our AI analysis has uncovered the following key insights:

"""
    
    insights = results.get('insights', [])
    if insights:
        for i, insight in enumerate(insights, 1):
            report += f"**{i}.** {insight}\n\n"
    else:
        report += "*No specific insights were generated for this dataset.*\n\n"
    
    report += """---

## 🎯 Recommended Actions

Based on the data analysis, we recommend the following strategic actions:

"""
    
    recommendations = results.get('recommendations', [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            report += f"**{i}.** {rec}\n\n"
    else:
        report += "*No specific recommendations were generated for this dataset.*\n\n"
    
    report += f"""---

## 🔧 Technical Summary

- **Analysis Completed:** {results.get('analysis_timestamp', 'N/A')}
- **Visualizations Created:** {len(results.get('visualizations', []))}
- **Processing Errors:** {len(results.get('errors', []))}
- **AI Model Used:** Llama 3 (70B parameters)

---

## 📈 Next Steps

1. **Review Insights:** Analyze each insight for immediate actionable opportunities
2. **Implement Recommendations:** Prioritize recommendations based on business impact
3. **Monitor Progress:** Track key metrics identified in this analysis
4. **Iterate:** Regular re-analysis as new data becomes available

---

*This report was generated automatically by our AI Data Analysis Agent. For questions or support, please contact your data team.*
"""
    
    return report

def main():
    """Main application function with beautiful design"""
    initialize_session_state()
    
    # Check if analysis is complete to show results immediately
    if st.session_state.analysis_complete and st.session_state.analysis_results:
        display_results()
        
        # Add a "Start New Analysis" button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Start New Analysis", use_container_width=True):
                # Reset session state
                st.session_state.analysis_results = None
                st.session_state.analysis_complete = False
                st.session_state.dataset = None
                st.rerun()
        return
    
    # Hero Section
    display_hero_section()
    
    # Feature showcase
    display_features()
    
    # Sidebar configuration
    api_configured = sidebar_config()
    
    if not api_configured:
        # Beautiful warning with setup instructions
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border: 2px solid #f59e0b;
            border-radius: 16px;
            padding: 2rem;
            margin: 2rem 0;
            text-align: center;
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🔑</div>
            <h3 style="color: #92400e; margin-bottom: 1rem;">API Key Required</h3>
            <p style="color: #78350f; margin-bottom: 1.5rem;">
                Please configure your Groq API key to unlock the power of AI analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Expandable setup guide
        with st.expander("🚀 Quick Setup Guide", expanded=True):
            st.markdown("""
            ### Option 1: Environment Variable (Recommended)
            ```bash
            export GROQ_API_KEY="your_api_key_here"
            streamlit run web_app.py
            ```
            
            ### Option 2: Manual Entry
            1. Visit [Groq Console](https://console.groq.com/) 🔗
            2. Create a free account and generate your API key
            3. Enter the key in the sidebar ← 
            4. Upload your dataset and start analyzing!
            
            ### Supported File Formats
            - **CSV files** (.csv) - Most common format
            - **Excel files** (.xlsx, .xls) - Spreadsheet data
            - **JSON files** (.json) - Structured data
            
            ### Tips for Best Results
            - Ensure clean, well-structured data
            - Include meaningful column names
            - Mix of numeric and categorical columns works best
            - Date/time columns enable trend analysis
            """)
        return
    
    # Main content area with beautiful layout
    st.markdown("---")
    
    # Dataset upload section
    dataset_uploaded = upload_dataset()
    
    # Analysis section
    if dataset_uploaded:
        st.markdown("---")
        
        # Center the analyze button with beautiful styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "🚀 Analyze My Data with AI",
                type="primary",
                use_container_width=True,
                help="Start the AI-powered analysis of your dataset"
            ):
                run_analysis()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <div style="max-width: 800px; margin: 0 auto;">
            <div style="font-size: 1.5rem; margin-bottom: 1rem;">🤖✨</div>
            <p style="margin-bottom: 1rem;">
                <strong>AI Data Analysis Agent</strong> - Transform your data into actionable insights
            </p>
            <p style="font-size: 0.85rem; margin-bottom: 1rem;">
                Powered by <strong>Llama 3</strong> • Built with <strong>LangGraph</strong> • 
                Designed with <strong>Streamlit</strong>
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; font-size: 0.9rem;">
                <a href="#" style="color: #3b82f6; text-decoration: none;">📖 Documentation</a>
                <a href="#" style="color: #3b82f6; text-decoration: none;">🐛 Report Issues</a>
                <a href="#" style="color: #3b82f6; text-decoration: none;">⭐ Give Feedback</a>
                <a href="#" style="color: #3b82f6; text-decoration: none;">💡 Feature Requests</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()