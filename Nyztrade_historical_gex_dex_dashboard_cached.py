# ============================================================================
# NYZTrade Historical GEX/DEX Dashboard - WITH 5-YEAR DATA CACHE
# Features: Gamma Flip Zones + Local Data Cache for 5+ Years Historical Data
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from datetime import datetime, timedelta
import pytz
import requests
import time
from dataclasses import dataclass
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import cache manager
from data_cache_manager import DataCacheManager

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================

st.set_page_config(
    page_title="NYZTrade Historical GEX/DEX",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 🔑 DHAN API CREDENTIALS - UPDATE YOUR TOKEN HERE
# ============================================================================
# Get your access token from: https://dhanhq.co/
# Access tokens expire regularly, so update this when you get "No data" errors

DHAN_CLIENT_ID = "1100480354"
DHAN_ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzY3NTk3NzM2LCJhcHBfaWQiOiJlZDMwMzI5NCIsImlhdCI6MTc2NzUxMTMzNiwidG9rZW5Db25zdW1lclR5cGUiOiJBUFAiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMDQ4MDM1NCJ9.op6hoq4ci1ODkJiAic7a98p-7OUxCGnSgDpV5A8-mWNLq2vTyqKrVzElmeTNRdkbLGfDcVx1wYitNP3JUbRpRQ"

# ✅ Token updated: 2026-01-04 | Expires: 2026-01-05 12:52 PM IST (~24 hours)
# ⚠️ Token expired? Update DHAN_ACCESS_TOKEN above with your new token!
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Comprehensive GitHub icon hiding - Multiple approaches */
    
    /* Method 1: Hide by href */
    header a[href*="github"],
    [data-testid="stHeader"] a[href*="github"],
    a[href*="github.com"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
    }
    
    /* Method 2: Hide by aria-label */
    a[aria-label*="GitHub"],
    a[aria-label*="github"],
    button[aria-label*="GitHub"],
    button[aria-label*="github"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Method 3: Hide toolbar buttons with GitHub SVG */
    header button svg path[d*="M12"],
    [data-testid="stToolbar"] button svg,
    [data-testid="stHeader"] button svg {
        display: none !important;
    }
    
    /* Method 4: Hide the entire toolbar GitHub button */
    header div[data-testid="stToolbar"] > div > button:last-child,
    [data-testid="stToolbar"] > div > button:nth-last-child(2) {
        display: none !important;
    }
    
    /* Method 5: Target the GitHub icon container */
    .stApp header button[kind="header"] {
        display: none !important;
    }
    
    /* Method 6: Nuclear option - hide all header icon buttons except specific ones */
    header button:has(svg) {
        display: none !important;
    }
    
    /* But keep the menu button visible */
    header button[aria-label="Open menu"],
    header button[aria-label*="menu"],
    header button:has(svg path[d*="M3 12h18"]) {
        display: block !important;
    }
    
    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #111827;
        --bg-card: #1a2332;
        --bg-card-hover: #232f42;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --accent-yellow: #f59e0b;
        --accent-cyan: #06b6d4;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --border-color: #2d3748;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, #0f172a 50%, var(--bg-primary) 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .sub-title {
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 8px;
    }
    
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        background: var(--bg-card-hover);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card.positive { border-left: 4px solid var(--accent-green); }
    .metric-card.negative { border-left: 4px solid var(--accent-red); }
    .metric-card.neutral { border-left: 4px solid var(--accent-yellow); }
    
    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }
    
    .metric-value.positive { color: var(--accent-green); }
    .metric-value.negative { color: var(--accent-red); }
    .metric-value.neutral { color: var(--accent-yellow); }
    
    .metric-delta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        margin-top: 8px;
        color: var(--text-secondary);
    }
    
    .signal-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 20px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .signal-badge.bullish {
        background: rgba(16, 185, 129, 0.15);
        color: var(--accent-green);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .signal-badge.bearish {
        background: rgba(239, 68, 68, 0.15);
        color: var(--accent-red);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .signal-badge.volatile {
        background: rgba(245, 158, 11, 0.15);
        color: var(--accent-yellow);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .history-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 20px;
    }
    
    .history-dot {
        width: 8px;
        height: 8px;
        background: var(--accent-blue);
        border-radius: 50%;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DhanConfig:
    client_id: str = DHAN_CLIENT_ID
    access_token: str = DHAN_ACCESS_TOKEN

DHAN_SECURITY_IDS = {
    "NIFTY": 13, 
    "BANKNIFTY": 25, 
    "FINNIFTY": 27, 
    "MIDCPNIFTY": 442,
    "SENSEX": 51
}

SYMBOL_CONFIG = {
    "NIFTY": {"contract_size": 25, "strike_interval": 50},
    "BANKNIFTY": {"contract_size": 15, "strike_interval": 100},
    "FINNIFTY": {"contract_size": 40, "strike_interval": 50},
    "MIDCPNIFTY": {"contract_size": 75, "strike_interval": 25},
    "SENSEX": {"contract_size": 10, "strike_interval": 100},
}

IST = pytz.timezone('Asia/Kolkata')

# ============================================================================
# BLACK-SCHOLES CALCULATOR
# ============================================================================

class BlackScholesCalculator:
    @staticmethod
    def calculate_d1(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return 0
        return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def calculate_d2(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return 0
        d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
        return d1 - sigma * np.sqrt(T)
    
    @staticmethod
    def calculate_gamma(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            return norm.pdf(d1) / (S * sigma * np.sqrt(T))
        except:
            return 0
    
    @staticmethod
    def calculate_call_delta(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            return norm.cdf(d1)
        except:
            return 0
    
    @staticmethod
    def calculate_put_delta(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            return norm.cdf(d1) - 1
        except:
            return 0
    
    @staticmethod
    def calculate_vanna(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            d2 = BlackScholesCalculator.calculate_d2(S, K, T, r, sigma)
            vanna = -norm.pdf(d1) * d2 / sigma
            return vanna
        except:
            return 0
    
    @staticmethod
    def calculate_charm(S, K, T, r, sigma, option_type='call'):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            d2 = BlackScholesCalculator.calculate_d2(S, K, T, r, sigma)
            charm = -norm.pdf(d1) * (2*r*T - d2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T))
            return charm
        except:
            return 0

# ============================================================================
# GAMMA FLIP ZONE CALCULATOR - NEW ADDITION
# ============================================================================

# 🎨 CUSTOMIZE GAMMA FLIP COLORS HERE
GAMMA_FLIP_COLORS = {
    'amplification': '#ef4444',  # Red - for amplification zones (change as needed)
    'suppression': '#10b981',    # Green - for suppression zones (change as needed)
    'opacity': 0.15,              # Box transparency (0.0 = invisible, 1.0 = solid)
    'line_width': 2,              # Border line width
    'annotation_size': 10         # Text size for flip labels
}

# Popular color alternatives:
# Blue: '#3b82f6'
# Purple: '#8b5cf6'
# Orange: '#f59e0b'
# Cyan: '#06b6d4'
# Yellow: '#eab308'
# Pink: '#ec4899'

def identify_gamma_flip_zones(df: pd.DataFrame, spot_price: float) -> List[Dict]:
    """
    Identifies gamma flip zones where GEX crosses zero.
    Returns list of flip zones with strike levels and direction indicators.
    """
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    
    flip_zones = []
    
    for i in range(len(df_sorted) - 1):
        current_gex = df_sorted.iloc[i]['net_gex']
        next_gex = df_sorted.iloc[i + 1]['net_gex']
        current_strike = df_sorted.iloc[i]['strike']
        next_strike = df_sorted.iloc[i + 1]['strike']
        
        # Check if GEX crosses zero between these strikes
        if (current_gex > 0 and next_gex < 0) or (current_gex < 0 and next_gex > 0):
            # Interpolate the exact flip strike
            flip_strike = current_strike + (next_strike - current_strike) * (abs(current_gex) / (abs(current_gex) + abs(next_gex)))
            
            # Determine flip direction based on spot position
            if spot_price < flip_strike:
                # Spot is below flip zone
                if current_gex > 0:
                    direction = "upward"  # Moving up crosses from positive to negative (suppression to amplification)
                    arrow = "↑"
                    color = GAMMA_FLIP_COLORS['amplification']  # Red - amplification above
                else:
                    direction = "downward"
                    arrow = "↓"
                    color = GAMMA_FLIP_COLORS['suppression']
            else:
                # Spot is above flip zone
                if current_gex < 0:
                    direction = "downward"  # Moving down crosses from negative to positive (amplification to suppression)
                    arrow = "↓"
                    color = GAMMA_FLIP_COLORS['suppression']  # Green - suppression below
                else:
                    direction = "upward"
                    arrow = "↑"
                    color = GAMMA_FLIP_COLORS['amplification']
            
            flip_zones.append({
                'strike': flip_strike,
                'lower_strike': current_strike,
                'upper_strike': next_strike,
                'lower_gex': current_gex,
                'upper_gex': next_gex,
                'direction': direction,
                'arrow': arrow,
                'color': color,
                'flip_type': 'Positive→Negative' if current_gex > 0 else 'Negative→Positive'
            })
    
    return flip_zones

# ============================================================================
# DHAN ROLLING API FETCHER
# ============================================================================

class DhanHistoricalFetcher:
    def __init__(self, config: DhanConfig):
        self.config = config
        self.headers = {
            'access-token': config.access_token,
            'client-id': config.client_id,
            'Content-Type': 'application/json'
        }
        self.base_url = "https://api.dhan.co/v2"
        self.bs_calc = BlackScholesCalculator()
        self.risk_free_rate = 0.07
    
    def fetch_rolling_data(self, symbol: str, from_date: str, to_date: str, 
                          strike_type: str = "ATM", option_type: str = "CALL", 
                          interval: str = "60", expiry_code: int = 1, expiry_flag: str = "WEEK"):
        try:
            security_id = DHAN_SECURITY_IDS.get(symbol, 13)
            
            payload = {
                "exchangeSegment": "NSE_FNO",
                "interval": interval,
                "securityId": security_id,
                "instrument": "OPTIDX",
                "expiryFlag": expiry_flag,
                "expiryCode": expiry_code,
                "strike": strike_type,
                "drvOptionType": option_type,
                "requiredData": ["open", "high", "low", "close", "volume", "oi", "iv", "strike", "spot"],
                "fromDate": from_date,
                "toDate": to_date
            }
            
            response = requests.post(
                f"{self.base_url}/charts/rollingoption",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            # Debug logging
            if response.status_code != 200:
                st.error(f"⚠️ API Status Code: {response.status_code}")
                try:
                    error_data = response.json()
                    st.error(f"API Error: {error_data}")
                except:
                    st.error(f"Response Text: {response.text[:200]}")
            
            if response.status_code == 200:
                data = response.json().get('data', {})
                
                # Debug: Check if data is empty
                if not data or not data.get('ce') or not data.get('pe'):
                    st.warning(f"⚠️ Empty data returned for {strike_type} {option_type}")
                    st.info(f"Date range: {from_date} to {to_date}")
                    return None
                
                return data
            elif response.status_code == 401:
                # Token expired or invalid
                st.error("🔑 **ACCESS TOKEN EXPIRED!** Please update your Dhan access token at the top of the code.")
                st.info("Get a new token from: https://dhanhq.co/ → API Keys section")
                return None
            else:
                st.warning(f"API returned status code: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    def process_historical_data(self, symbol: str, target_date: str, strikes: List[str], 
                               interval: str = "60", expiry_code: int = 1, expiry_flag: str = "WEEK"):
        
        target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        
        from_date = (target_dt - timedelta(days=2)).strftime('%Y-%m-%d')
        to_date = (target_dt + timedelta(days=2)).strftime('%Y-%m-%d')
        
        config = SYMBOL_CONFIG.get(symbol, SYMBOL_CONFIG["NIFTY"])
        contract_size = config["contract_size"]
        
        all_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_steps = len(strikes) * 2
        current_step = 0
        
        for strike_type in strikes:
            status_text.text(f"Fetching {strike_type} ({expiry_flag} Expiry Code: {expiry_code})...")
            
            call_data = self.fetch_rolling_data(symbol, from_date, to_date, strike_type, "CALL", interval, expiry_code, expiry_flag)
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            time.sleep(1)
            
            put_data = self.fetch_rolling_data(symbol, from_date, to_date, strike_type, "PUT", interval, expiry_code, expiry_flag)
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            time.sleep(1)
            
            if not call_data or not put_data:
                continue
            
            ce_data = call_data.get('ce', {})
            pe_data = put_data.get('pe', {})
            
            if not ce_data:
                continue
            
            timestamps = ce_data.get('timestamp', [])
            
            for i, ts in enumerate(timestamps):
                try:
                    dt_utc = datetime.fromtimestamp(ts, tz=pytz.UTC)
                    dt_ist = dt_utc.astimezone(IST)
                    
                    if dt_ist.date() != target_dt.date():
                        continue
                    
                    spot_price = ce_data.get('spot', [0])[i] if i < len(ce_data.get('spot', [])) else 0
                    strike_price = ce_data.get('strike', [0])[i] if i < len(ce_data.get('strike', [])) else 0
                    
                    if spot_price == 0 or strike_price == 0:
                        continue
                    
                    call_oi = ce_data.get('oi', [0])[i] if i < len(ce_data.get('oi', [])) else 0
                    put_oi = pe_data.get('oi', [0])[i] if i < len(pe_data.get('oi', [])) else 0
                    call_volume = ce_data.get('volume', [0])[i] if i < len(ce_data.get('volume', [])) else 0
                    put_volume = pe_data.get('volume', [0])[i] if i < len(pe_data.get('volume', [])) else 0
                    call_iv = ce_data.get('iv', [15])[i] if i < len(ce_data.get('iv', [])) else 15
                    put_iv = pe_data.get('iv', [15])[i] if i < len(pe_data.get('iv', [])) else 15
                    
                    time_to_expiry = 7 / 365
                    call_iv_dec = call_iv / 100 if call_iv > 1 else call_iv
                    put_iv_dec = put_iv / 100 if put_iv > 1 else put_iv
                    
                    call_gamma = self.bs_calc.calculate_gamma(spot_price, strike_price, time_to_expiry, self.risk_free_rate, call_iv_dec)
                    put_gamma = self.bs_calc.calculate_gamma(spot_price, strike_price, time_to_expiry, self.risk_free_rate, put_iv_dec)
                    call_delta = self.bs_calc.calculate_call_delta(spot_price, strike_price, time_to_expiry, self.risk_free_rate, call_iv_dec)
                    put_delta = self.bs_calc.calculate_put_delta(spot_price, strike_price, time_to_expiry, self.risk_free_rate, put_iv_dec)
                    
                    call_vanna = self.bs_calc.calculate_vanna(spot_price, strike_price, time_to_expiry, self.risk_free_rate, call_iv_dec)
                    put_vanna = self.bs_calc.calculate_vanna(spot_price, strike_price, time_to_expiry, self.risk_free_rate, put_iv_dec)
                    call_charm = self.bs_calc.calculate_charm(spot_price, strike_price, time_to_expiry, self.risk_free_rate, call_iv_dec, 'call')
                    put_charm = self.bs_calc.calculate_charm(spot_price, strike_price, time_to_expiry, self.risk_free_rate, put_iv_dec, 'put')
                    
                    call_gex = (call_oi * call_gamma * spot_price**2 * contract_size) / 1e9
                    put_gex = -(put_oi * put_gamma * spot_price**2 * contract_size) / 1e9
                    call_dex = (call_oi * call_delta * spot_price * contract_size) / 1e9
                    put_dex = (put_oi * put_delta * spot_price * contract_size) / 1e9
                    
                    call_vanna_exp = (call_oi * call_vanna * spot_price * contract_size) / 1e9
                    put_vanna_exp = (put_oi * put_vanna * spot_price * contract_size) / 1e9
                    call_charm_exp = (call_oi * call_charm * spot_price * contract_size) / 1e9
                    put_charm_exp = (put_oi * put_charm * spot_price * contract_size) / 1e9
                    
                    all_data.append({
                        'timestamp': dt_ist,
                        'time': dt_ist.strftime('%H:%M IST'),
                        'spot_price': spot_price,
                        'strike': strike_price,
                        'strike_type': strike_type,
                        'call_oi': call_oi,
                        'put_oi': put_oi,
                        'call_volume': call_volume,
                        'put_volume': put_volume,
                        'total_volume': call_volume + put_volume,
                        'call_iv': call_iv,
                        'put_iv': put_iv,
                        'call_gex': call_gex,
                        'put_gex': put_gex,
                        'net_gex': call_gex + put_gex,
                        'call_dex': call_dex,
                        'put_dex': put_dex,
                        'net_dex': call_dex + put_dex,
                        'call_vanna': call_vanna_exp,
                        'put_vanna': put_vanna_exp,
                        'net_vanna': call_vanna_exp + put_vanna_exp,
                        'call_charm': call_charm_exp,
                        'put_charm': put_charm_exp,
                        'net_charm': call_charm_exp + put_charm_exp,
                    })
                    
                except Exception as e:
                    continue
        
        progress_bar.empty()
        status_text.empty()
        
        if not all_data:
            return None, None
        
        df = pd.DataFrame(all_data)
        df = df.sort_values(['strike', 'timestamp']).reset_index(drop=True)
        
        df['call_gex_flow'] = 0.0
        df['put_gex_flow'] = 0.0
        df['net_gex_flow'] = 0.0
        df['call_dex_flow'] = 0.0
        df['put_dex_flow'] = 0.0
        df['net_dex_flow'] = 0.0
        
        for strike in df['strike'].unique():
            strike_mask = df['strike'] == strike
            strike_data = df[strike_mask].copy()
            
            if len(strike_data) > 1:
                df.loc[strike_mask, 'call_gex_flow'] = strike_data['call_gex'].diff().fillna(0)
                df.loc[strike_mask, 'put_gex_flow'] = strike_data['put_gex'].diff().fillna(0)
                df.loc[strike_mask, 'net_gex_flow'] = strike_data['net_gex'].diff().fillna(0)
                df.loc[strike_mask, 'call_dex_flow'] = strike_data['call_dex'].diff().fillna(0)
                df.loc[strike_mask, 'put_dex_flow'] = strike_data['put_dex'].diff().fillna(0)
                df.loc[strike_mask, 'net_dex_flow'] = strike_data['net_dex'].diff().fillna(0)
        
        max_gex = df['net_gex'].abs().max()
        df['hedging_pressure'] = (df['net_gex'] / max_gex * 100) if max_gex > 0 else 0
        
        latest = df.sort_values('timestamp').iloc[-1]
        spot_prices = df['spot_price'].unique()
        spot_variation = (spot_prices.max() - spot_prices.min()) / spot_prices.mean() * 100
        
        meta = {
            'symbol': symbol,
            'date': target_date,
            'spot_price': latest['spot_price'],
            'spot_price_min': spot_prices.min(),
            'spot_price_max': spot_prices.max(),
            'spot_variation_pct': spot_variation,
            'total_records': len(df),
            'time_range': f"{df['time'].min()} - {df['time'].max()}",
            'strikes_count': df['strike'].nunique(),
            'interval': f"{interval} minutes",
            'expiry_code': expiry_code,
            'expiry_flag': expiry_flag
        }
        
        return df, meta

# ============================================================================
# CHART FUNCTIONS - ORIGINAL WITH GAMMA FLIP ZONES ADDED
# ============================================================================

def create_intraday_timeline(df: pd.DataFrame, selected_timestamp) -> go.Figure:
    timeline_df = df.groupby('timestamp').agg({
        'net_gex': 'sum',
        'net_dex': 'sum',
        'spot_price': 'first'
    }).reset_index()
    
    timeline_df = timeline_df.sort_values('timestamp')
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Total Net GEX Over Time', 'Total Net DEX Over Time', 'Spot Price Movement'),
        vertical_spacing=0.1,
        row_heights=[0.35, 0.35, 0.3]
    )
    
    gex_colors = ['#10b981' if x > 0 else '#ef4444' for x in timeline_df['net_gex']]
    fig.add_trace(
        go.Bar(
            x=timeline_df['timestamp'],
            y=timeline_df['net_gex'],
            marker_color=gex_colors,
            name='Net GEX',
            hovertemplate='%{x|%H:%M}<br>GEX: %{y:.4f}B<extra></extra>'
        ),
        row=1, col=1
    )
    
    dex_colors = ['#10b981' if x > 0 else '#ef4444' for x in timeline_df['net_dex']]
    fig.add_trace(
        go.Bar(
            x=timeline_df['timestamp'],
            y=timeline_df['net_dex'],
            marker_color=dex_colors,
            name='Net DEX',
            hovertemplate='%{x|%H:%M}<br>DEX: %{y:.4f}B<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=timeline_df['timestamp'],
            y=timeline_df['spot_price'],
            mode='lines+markers',
            line=dict(color='#3b82f6', width=2),
            marker=dict(size=4),
            name='Spot Price',
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)',
            hovertemplate='%{x|%H:%M}<br>Spot: ₹%{y:,.2f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    fig.add_vline(x=selected_timestamp, line_dash="dash", line_color="#f59e0b", line_width=3, row=1, col=1)
    fig.add_vline(x=selected_timestamp, line_dash="dash", line_color="#f59e0b", line_width=3, row=2, col=1)
    fig.add_vline(x=selected_timestamp, line_dash="dash", line_color="#f59e0b", line_width=3, row=3, col=1)
    
    fig.update_layout(
        title=dict(text="<b>📈 Intraday Evolution</b>", font=dict(size=18, color='white')),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=900,
        showlegend=False,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Time (IST)", row=3, col=1)
    fig.update_yaxes(title_text="GEX (₹B)", row=1, col=1)
    fig.update_yaxes(title_text="DEX (₹B)", row=2, col=1)
    fig.update_yaxes(title_text="Spot Price (₹)", row=3, col=1)
    
    return fig

def create_net_gex_flow_chart(df: pd.DataFrame, spot_price: float) -> go.Figure:
    """NET GEX Flow chart"""
    fig = go.Figure()
    
    colors = ['#10b981' if x > 0 else '#ef4444' for x in df['net_gex_flow']]
    
    fig.add_trace(go.Bar(
        y=df['strike'],
        x=df['net_gex_flow'],
        orientation='h',
        name='NET GEX Flow',
        marker_color=colors,
        hovertemplate='Strike: %{y}<br>NET GEX Flow: %{x:.4f}B<extra></extra>'
    ))
    
    fig.add_hline(
        y=spot_price,
        line_dash="dash",
        line_color="#3b82f6",
        line_width=2,
        annotation_text=f"Spot: ₹{spot_price:,.0f}",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=dict(text="<b>🌊 NET GEX Flow Distribution</b>", font=dict(size=18, color='white')),
        xaxis_title="NET GEX Flow (₹B)",
        yaxis_title="Strike Price",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=600,
        hovermode='y unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_net_dex_flow_chart(df: pd.DataFrame, spot_price: float) -> go.Figure:
    """NET DEX Flow chart"""
    fig = go.Figure()
    
    colors = ['#10b981' if x > 0 else '#ef4444' for x in df['net_dex_flow']]
    
    fig.add_trace(go.Bar(
        y=df['strike'],
        x=df['net_dex_flow'],
        orientation='h',
        name='NET DEX Flow',
        marker_color=colors,
        hovertemplate='Strike: %{y}<br>NET DEX Flow: %{x:.4f}B<extra></extra>'
    ))
    
    fig.add_hline(
        y=spot_price,
        line_dash="dash",
        line_color="#3b82f6",
        line_width=2,
        annotation_text=f"Spot: ₹{spot_price:,.0f}",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=dict(text="<b>🌊 NET DEX Flow Distribution</b>", font=dict(size=18, color='white')),
        xaxis_title="NET DEX Flow (₹B)",
        yaxis_title="Strike Price",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=600,
        hovermode='y unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_separate_gex_chart(df: pd.DataFrame, spot_price: float) -> go.Figure:
    """GEX chart - WITH GAMMA FLIP ZONES ADDED"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    colors = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['net_gex']]
    
    # NEW: Identify gamma flip zones
    flip_zones = identify_gamma_flip_zones(df_sorted, spot_price)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['net_gex'],
        orientation='h',
        marker_color=colors,
        name='Net GEX',
        hovertemplate='Strike: %{y:,.0f}<br>Net GEX: %{x:.4f}B<extra></extra>',
        showlegend=False
    ))
    
    fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=3,
                  annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right",
                  annotation=dict(font=dict(size=12, color="white")))
    
    # NEW: Add gamma flip zones
    for zone in flip_zones:
        # Add horizontal line at flip zone
        fig.add_hline(
            y=zone['strike'],
            line_dash="dot",
            line_color=zone['color'],
            line_width=2,
            annotation_text=f"🔄 Flip {zone['arrow']} {zone['strike']:,.0f}",
            annotation_position="left",
            annotation=dict(
                font=dict(size=GAMMA_FLIP_COLORS['annotation_size'], color=zone['color']),
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor=zone['color'],
                borderwidth=1
            )
        )
        
        # Add shaded region around flip zone
        fig.add_hrect(
            y0=zone['lower_strike'],
            y1=zone['upper_strike'],
            fillcolor=zone['color'],
            opacity=GAMMA_FLIP_COLORS['opacity'],
            line_width=0,
            annotation_text=zone['arrow'],
            annotation_position="right",
            annotation=dict(font=dict(size=16, color=zone['color']))
        )
    
    fig.update_layout(
        title=dict(text="<b>🎯 Gamma Exposure (GEX) with Flip Zones</b>", font=dict(size=18, color='white')),
        xaxis_title="GEX (₹ Billions)",
        yaxis_title="Strike Price",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=700,
        showlegend=False,
        hovermode='closest',
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True, autorange=True),
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig

def create_separate_dex_chart(df: pd.DataFrame, spot_price: float) -> go.Figure:
    """DEX chart - ORIGINAL"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    colors = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['net_dex']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['net_dex'],
        orientation='h',
        marker_color=colors,
        name='Net DEX',
        hovertemplate='Strike: %{y:,.0f}<br>Net DEX: %{x:.4f}B<extra></extra>',
        showlegend=False
    ))
    
    fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=3,
                  annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right",
                  annotation=dict(font=dict(size=12, color="white")))
    
    fig.update_layout(
        title=dict(text="<b>📊 Delta Exposure (DEX)</b>", font=dict(size=18, color='white')),
        xaxis_title="DEX (₹ Billions)",
        yaxis_title="Strike Price",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=700,
        showlegend=False,
        hovermode='closest',
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True, autorange=True),
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig

def create_net_gex_dex_chart(df: pd.DataFrame, spot_price: float) -> go.Figure:
    """NET GEX+DEX chart - WITH GAMMA FLIP ZONES ADDED"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    df_sorted['net_gex_dex'] = df_sorted['net_gex'] + df_sorted['net_dex']
    colors = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['net_gex_dex']]
    
    # NEW: Identify gamma flip zones
    flip_zones = identify_gamma_flip_zones(df_sorted, spot_price)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['net_gex_dex'],
        orientation='h',
        marker_color=colors,
        name='Net GEX+DEX',
        hovertemplate='Strike: %{y:,.0f}<br>Net GEX+DEX: %{x:.4f}B<extra></extra>',
        showlegend=False
    ))
    
    fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=3,
                  annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right",
                  annotation=dict(font=dict(size=12, color="white")))
    
    # NEW: Add gamma flip zones
    for zone in flip_zones:
        fig.add_hline(
            y=zone['strike'],
            line_dash="dot",
            line_color=zone['color'],
            line_width=2,
            annotation_text=f"🔄 Flip {zone['arrow']} {zone['strike']:,.0f}",
            annotation_position="left",
            annotation=dict(
                font=dict(size=GAMMA_FLIP_COLORS['annotation_size'], color=zone['color']),
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor=zone['color'],
                borderwidth=1
            )
        )
        
        fig.add_hrect(
            y0=zone['lower_strike'],
            y1=zone['upper_strike'],
            fillcolor=zone['color'],
            opacity=GAMMA_FLIP_COLORS['opacity'],
            line_width=0,
            annotation_text=zone['arrow'],
            annotation_position="right",
            annotation=dict(font=dict(size=16, color=zone['color']))
        )
    
    fig.update_layout(
        title=dict(text="<b>⚡ Combined NET GEX + DEX with Flip Zones</b>", font=dict(size=18, color='white')),
        xaxis_title="Combined Exposure (₹ Billions)",
        yaxis_title="Strike Price",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=700,
        showlegend=False,
        hovermode='closest',
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True, autorange=True),
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig

def create_hedging_pressure_chart(df: pd.DataFrame, spot_price: float) -> go.Figure:
    """Hedging pressure chart - WITH GAMMA FLIP ZONES ADDED"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    
    # NEW: Identify gamma flip zones
    flip_zones = identify_gamma_flip_zones(df_sorted, spot_price)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['hedging_pressure'],
        orientation='h',
        marker=dict(
            color=df_sorted['hedging_pressure'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(
                title=dict(text='Pressure %', font=dict(color='white', size=12)),
                tickfont=dict(color='white'),
                x=1.02,
                len=0.7,
                thickness=20
            ),
            cmin=-100,
            cmax=100
        ),
        hovertemplate='Strike: %{y:,.0f}<br>Pressure: %{x:.1f}%<extra></extra>',
        showlegend=False
    ))
    
    fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=3,
                  annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right",
                  annotation=dict(font=dict(size=12, color="white")))
    
    fig.add_vline(x=0, line_dash="dot", line_color="gray", line_width=1)
    
    # NEW: Add gamma flip zones
    for zone in flip_zones:
        fig.add_hline(
            y=zone['strike'],
            line_dash="dot",
            line_color=zone['color'],
            line_width=2,
            annotation_text=f"🔄 Flip {zone['arrow']} {zone['strike']:,.0f}",
            annotation_position="left",
            annotation=dict(
                font=dict(size=GAMMA_FLIP_COLORS['annotation_size'], color=zone['color']),
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor=zone['color'],
                borderwidth=1
            )
        )
        
        fig.add_hrect(
            y0=zone['lower_strike'],
            y1=zone['upper_strike'],
            fillcolor=zone['color'],
            opacity=GAMMA_FLIP_COLORS['opacity'],
            line_width=0,
            annotation_text=zone['arrow'],
            annotation_position="right",
            annotation=dict(font=dict(size=16, color=zone['color']))
        )
    
    fig.update_layout(
        title=dict(text="<b>🎪 Hedging Pressure with Flip Zones</b>", font=dict(size=18, color='white')),
        xaxis_title="Hedging Pressure (%)",
        yaxis_title="Strike Price",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=700,
        showlegend=False,
        hovermode='closest',
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)', 
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(128,128,128,0.5)',
            zerolinewidth=2,
            range=[-110, 110]
        ),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True, autorange=True),
        margin=dict(l=80, r=120, t=80, b=80)
    )
    
    return fig

def create_oi_distribution(df: pd.DataFrame, spot_price: float) -> go.Figure:
    """OI Distribution - ORIGINAL"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['call_oi'],
        orientation='h',
        name='Call OI',
        marker_color='#10b981',
        opacity=0.7,
        hovertemplate='Strike: %{y:,.0f}<br>Call OI: %{x:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=-df_sorted['put_oi'],
        orientation='h',
        name='Put OI',
        marker_color='#ef4444',
        opacity=0.7,
        hovertemplate='Strike: %{y:,.0f}<br>Put OI: %{customdata:,.0f}<extra></extra>',
        customdata=df_sorted['put_oi']
    ))
    
    fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=2,
                  annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right",
                  annotation=dict(font=dict(size=12, color="white")))
    
    fig.add_vline(x=0, line_dash="dot", line_color="white", line_width=1)
    
    fig.update_layout(
        title=dict(text="<b>📋 Open Interest Distribution</b>", font=dict(size=16, color='white')),
        xaxis_title="Open Interest (Calls +ve | Puts -ve)",
        yaxis_title="Strike Price",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=500,
        barmode='overlay',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color='white')),
        hovermode='closest',
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)', 
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.3)',
            zerolinewidth=2
        ),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True, autorange=True),
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig

def create_vanna_exposure_chart(df: pd.DataFrame, spot_price: float) -> go.Figure:
    """VANNA Exposure - ORIGINAL"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    
    colors_call = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['call_vanna']]
    colors_put = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['put_vanna']]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("📈 Call VANNA", "📉 Put VANNA"),
        horizontal_spacing=0.12
    )
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['call_vanna'],
        orientation='h',
        marker=dict(color=colors_call),
        name='Call VANNA',
        hovertemplate='Strike: %{y:,.0f}<br>Call VANNA: %{x:.4f}B<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['put_vanna'],
        orientation='h',
        marker=dict(color=colors_put),
        name='Put VANNA',
        hovertemplate='Strike: %{y:,.0f}<br>Put VANNA: %{x:.4f}B<extra></extra>'
    ), row=1, col=2)
    
    for col in [1, 2]:
        fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=2,
                      annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right",
                      annotation=dict(font=dict(size=10, color="white")), row=1, col=col)
    
    fig.update_layout(
        title=dict(text="<b>🌊 VANNA Exposure (dDelta/dVol)</b>", font=dict(size=18, color='white')),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=600,
        showlegend=False,
        hovermode='closest',
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    fig.update_xaxes(title_text="VANNA (₹ Billions)", gridcolor='rgba(128,128,128,0.2)', showgrid=True)
    fig.update_yaxes(title_text="Strike Price", gridcolor='rgba(128,128,128,0.2)', showgrid=True)
    
    return fig

def create_charm_exposure_chart(df: pd.DataFrame, spot_price: float) -> go.Figure:
    """CHARM Exposure - ORIGINAL"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    
    colors_call = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['call_charm']]
    colors_put = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['put_charm']]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("📈 Call CHARM", "📉 Put CHARM"),
        horizontal_spacing=0.12
    )
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['call_charm'],
        orientation='h',
        marker=dict(color=colors_call),
        name='Call CHARM',
        hovertemplate='Strike: %{y:,.0f}<br>Call CHARM: %{x:.4f}B<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['put_charm'],
        orientation='h',
        marker=dict(color=colors_put),
        name='Put CHARM',
        hovertemplate='Strike: %{y:,.0f}<br>Put CHARM: %{x:.4f}B<extra></extra>'
    ), row=1, col=2)
    
    for col in [1, 2]:
        fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=2,
                      annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right",
                      annotation=dict(font=dict(size=10, color="white")), row=1, col=col)
    
    fig.update_layout(
        title=dict(text="<b>⏰ CHARM Exposure (Delta Decay)</b>", font=dict(size=18, color='white')),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=600,
        showlegend=False,
        hovermode='closest',
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    fig.update_xaxes(title_text="CHARM (₹ Billions)", gridcolor='rgba(128,128,128,0.2)', showgrid=True)
    fig.update_yaxes(title_text="Strike Price", gridcolor='rgba(128,128,128,0.2)', showgrid=True)
    
    return fig

# ============================================================================
# MAIN APPLICATION - ORIGINAL WITH FLIP ZONE INFO ADDED
# ============================================================================

def main():
    # Initialize cache manager
    cache_manager = DataCacheManager(cache_dir="data_cache")
    
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 class="main-title">📊 NYZTrade Historical GEX/DEX Dashboard</h1>
                <p class="sub-title">Historical Options Greeks Analysis | Gamma Flip Zones | 5-Year Data Cache | Indian Standard Time</p>
            </div>
            <div class="history-indicator">
                <div class="history-dot"></div>
                <span style="color: #3b82f6; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">CACHED DATA</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Check token expiry
        try:
            import jwt
            import time
            token_data = jwt.decode(DHAN_ACCESS_TOKEN, options={"verify_signature": False})
            exp_timestamp = token_data.get('exp', 0)
            current_timestamp = time.time()
            
            if exp_timestamp < current_timestamp:
                st.error("⚠️ **TOKEN EXPIRED!**")
                st.info("Update token in code (top of file)")
            elif exp_timestamp < (current_timestamp + 86400):  # Less than 24 hours
                st.warning("⚠️ Token expires soon")
            else:
                st.success("✅ Token valid")
        except:
            st.info("🔑 Token status: Unknown")
        
        st.markdown("---")
        
        symbol = st.selectbox(
            "📈 Select Index",
            options=list(DHAN_SECURITY_IDS.keys()),
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 📅 Historical Date Selection")
        
        live_mode = st.checkbox("📡 **LIVE DATA MODE** (Today's trading)", value=False)
        
        if live_mode:
            st.markdown("---")
            st.markdown("### 🔄 Auto-Refresh Settings")
            
            auto_refresh = st.checkbox("⚡ **Enable Auto-Refresh**", value=False)
            
            if auto_refresh:
                refresh_interval = st.slider(
                    "Refresh Interval (seconds)",
                    min_value=60,
                    max_value=180,
                    value=120,
                    step=30
                )
                
                quiet_mode = st.checkbox("🔇 Quiet Mode", value=True)
                
                st.success(f"🔄 Auto-refresh: **ON** | Every **{refresh_interval} seconds**")
                
                if quiet_mode:
                    st.info("💡 Quiet mode enabled - Smooth countdown, minimal blinking")
                
                st.session_state.auto_refresh_enabled = True
                st.session_state.refresh_interval = refresh_interval
                st.session_state.quiet_mode = quiet_mode
            else:
                st.session_state.auto_refresh_enabled = False
            
            st.markdown("---")
        else:
            st.session_state.auto_refresh_enabled = False
        
        if live_mode:
            today = datetime.now(IST).date()
            selected_date = today
            target_date = today.strftime('%Y-%m-%d')
            st.success(f"🔴 LIVE MODE | Fetching real-time data for: **{target_date}**")
        else:
            date_range_option = st.selectbox(
                "Select Date Range",
                ["Last 30 Days", "Last 60 Days", "Last 90 Days", "Last 6 Months", "Custom Range"],
                index=0
            )
            
            if date_range_option == "Custom Range":
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=datetime.now() - timedelta(days=30),
                        max_value=datetime.now(),
                        min_value=datetime.now() - timedelta(days=180)
                    )
                with col2:
                    end_date = st.date_input(
                        "End Date",
                        value=datetime.now(),
                        max_value=datetime.now(),
                        min_value=start_date
                    )
                
                date_list = pd.date_range(start=start_date, end=end_date, freq='D')
                date_list = [d for d in date_list if d.weekday() < 5]
            else:
                if date_range_option == "Last 30 Days":
                    days_back = 30
                elif date_range_option == "Last 60 Days":
                    days_back = 60
                elif date_range_option == "Last 90 Days":
                    days_back = 90
                else:
                    days_back = 180
                
                date_list = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
                date_list = [d for d in date_list if d.weekday() < 5]
            
            available_dates = [d.date() for d in date_list]
            
            if len(available_dates) > 0:
                st.caption(f"📊 {len(available_dates)} trading days available")
            
            selected_date = st.selectbox(
                "Select Trading Day",
                options=available_dates,
                index=len(available_dates)-1 if len(available_dates) > 0 else 0,
                format_func=lambda x: x.strftime('%Y-%m-%d (%A)')
            )
            
            target_date = selected_date.strftime('%Y-%m-%d')
            
            # Show data availability hint
            days_ago = (datetime.now().date() - selected_date).days
            if days_ago > 30:
                st.warning(f"⚠️ Selected date is {days_ago} days old. Data may not be available.")
                st.caption("💡 Best results: Last 7-14 days")
            elif days_ago > 14:
                st.info(f"📅 {days_ago} days ago - Data might be limited")
            elif days_ago <= 1:
                st.warning("⚠️ Very recent date - Historical data may not be ready yet")
                st.caption("💡 Try 2-3 days ago for reliable data")
        
        st.markdown("---")
        st.markdown("### 📆 Expiry Type & Selection")
        
        expiry_type = st.selectbox("Expiry Type", ["Weekly", "Monthly"], index=0)
        expiry_flag = "WEEK" if expiry_type == "Weekly" else "MONTH"
        
        expiry_option = st.selectbox(
            "Select Expiry",
            ["Current Week/Month (Nearest)", "Next Week/Month", "Far Week/Month"],
            index=0
        )
        
        expiry_code_map = {
            "Current Week/Month (Nearest)": 1,
            "Next Week/Month": 2,
            "Far Week/Month": 3
        }
        expiry_code = expiry_code_map[expiry_option]
        
        st.markdown("---")
        st.markdown("### 🎯 Strike Selection")
        
        strikes = st.multiselect(
            "Select Strikes",
            ["ATM", "ATM+1", "ATM-1", "ATM+2", "ATM-2", "ATM+3", "ATM-3", 
             "ATM+4", "ATM-4", "ATM+5", "ATM-5", "ATM+6", "ATM-6", "ATM+7", "ATM-7",
             "ATM+8", "ATM-8", "ATM+9", "ATM-9", "ATM+10", "ATM-10"],
            default=["ATM", "ATM+1", "ATM-1", "ATM+2", "ATM-2", "ATM+3", "ATM-3"]
        )
        
        st.markdown("---")
        st.markdown("### ⏱️ Time Interval")
        
        interval = st.selectbox(
            "Select Interval",
            options=["5", "15", "60"],
            format_func=lambda x: "5 minutes" if x == "5" else "15 minutes" if x == "15" else "1 hour",
            index=0
        )
        
        st.info(f"📊 Selected: {len(strikes)} strikes | {interval} min interval")
        
        st.markdown("---")
        
        # Cache status
        st.markdown("### 💾 Cache Status")
        cache_stats = cache_manager.get_cache_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cached Entries", f"{cache_stats['total_entries']}")
        with col2:
            st.metric("Cache Size", f"{cache_stats['total_size_mb']:.1f} MB")
        
        if cache_stats['total_entries'] > 0:
            st.caption(f"📅 Date range: {cache_stats['oldest_date']} to {cache_stats['newest_date']}")
            
            # Check if current selection is cached
            is_cached = cache_manager.is_cached(symbol, target_date, interval, expiry_code, expiry_flag, strikes)
            if is_cached:
                st.success("✅ This data is cached - instant load!")
            else:
                st.info("📡 Will fetch from API")
        
        st.markdown("---")
        
        fetch_button = st.button("🚀 Fetch Historical Data", use_container_width=True, type="primary")
        
        st.markdown("---")
        st.markdown("### 🕐 Indian Standard Time (IST)")
        ist_now = datetime.now(IST)
        st.info(f"Current IST: {ist_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    if fetch_button:
        st.session_state.fetch_config = {
            'symbol': symbol,
            'target_date': target_date,
            'strikes': strikes,
            'interval': interval,
            'expiry_code': expiry_code,
            'expiry_flag': expiry_flag
        }
        st.session_state.data_fetched = False
    
    if fetch_button or (hasattr(st.session_state, 'fetch_config') and st.session_state.get('data_fetched', False)):
        if hasattr(st.session_state, 'fetch_config'):
            config = st.session_state.fetch_config
            symbol = config['symbol']
            target_date = config['target_date']
            strikes = config['strikes']
            interval = config['interval']
            expiry_code = config.get('expiry_code', 1)
            expiry_flag = config.get('expiry_flag', 'WEEK')
        
        if not strikes:
            st.error("❌ Please select at least one strike")
            return
        
        if not st.session_state.get('data_fetched', False) or 'df_data' not in st.session_state:
            # First, check if data is in cache
            cached_data = cache_manager.load_from_cache(symbol, target_date, interval, expiry_code, expiry_flag, strikes)
            
            if cached_data is not None:
                df, meta = cached_data
                st.success("✅ Loaded from cache - instant!")
                
                st.session_state.df_data = df
                st.session_state.meta_data = meta
                st.session_state.data_fetched = True
                st.rerun()
            
            else:
                # Not in cache - fetch from API
                st.markdown(f"""
                <div class="metric-card neutral" style="margin: 20px 0;">
                    <div class="metric-label">Fetching from Dhan API (Not in Cache)</div>
                    <div class="metric-value" style="color: #3b82f6; font-size: 1.2rem;">
                        {symbol} | {target_date} | {interval} min | Strikes: {', '.join(strikes[:3])}...
                    </div>
                    <div class="metric-delta">This may take 1-3 minutes...</div>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    fetcher = DhanHistoricalFetcher(DhanConfig())
                    df, meta = fetcher.process_historical_data(symbol, target_date, strikes, interval, expiry_code, expiry_flag)
                
                if df is None or len(df) == 0:
                    st.error("❌ No data available for the selected date.")
                    st.info("""
                    **Troubleshooting Tips:**
                    
                    1. **Try a more recent date**: Dhan's historical data works best for the last 7-14 days
                       - Try: December 30, 2025 or later
                       - Current selection: """ + target_date + """
                    
                    2. **Verify it's a trading day**: 
                       - Market open Monday-Friday (except holidays)
                       - No trading on weekends or NSE holidays
                    
                    3. **For very recent dates** (yesterday/today):
                       - Historical data takes 1-2 days to become available
                       - Try enabling "Live Data Mode" instead
                    
                    4. **Check expiry selection**:
                       - Current Week expiry might not have data on all days
                       - Try "Next Week" or "Monthly" expiry
                    
                    5. **Token status**: Check sidebar for token validity
                    """)
                    return
                
                # Save to cache for future use
                cache_manager.save_to_cache(df, meta, symbol, target_date, interval, expiry_code, expiry_flag, strikes)
                st.success("✅ Data fetched and saved to cache!")
                
                st.session_state.df_data = df
                st.session_state.meta_data = meta
                st.session_state.data_fetched = True
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                return
        
        df = st.session_state.df_data
        meta = st.session_state.meta_data
        
        all_timestamps = sorted(df['timestamp'].unique())
        
        st.success(f"✅ Data fetched successfully! Total records: {len(df):,} | Strikes: {meta['strikes_count']}")
        
        st.markdown("---")
        st.markdown("### ⏱️ Time Navigation")
        
        control_cols = st.columns([1, 1, 1, 1, 1, 1, 1, 1])
        
        with control_cols[0]:
            if st.button("⏮️ First", use_container_width=True):
                st.session_state.timestamp_idx = 0
        
        with control_cols[1]:
            if st.button("◀️ Prev", use_container_width=True):
                current = st.session_state.get('timestamp_idx', len(all_timestamps) - 1)
                st.session_state.timestamp_idx = max(0, current - 1)
        
        with control_cols[2]:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.timestamp_idx = len(all_timestamps) - 1
        
        with control_cols[3]:
            if st.button("▶️ Next", use_container_width=True):
                current = st.session_state.get('timestamp_idx', len(all_timestamps) - 1)
                st.session_state.timestamp_idx = min(len(all_timestamps) - 1, current + 1)
        
        with control_cols[4]:
            if st.button("⏭️ Last", use_container_width=True):
                st.session_state.timestamp_idx = len(all_timestamps) - 1
        
        with control_cols[5]:
            if st.button("⏰ 9:30", use_container_width=True):
                morning_times = [i for i, ts in enumerate(all_timestamps) if ts.hour == 9 and ts.minute >= 30]
                if morning_times:
                    st.session_state.timestamp_idx = morning_times[0]
        
        with control_cols[6]:
            if st.button("⏰ 12:00", use_container_width=True):
                noon_times = [i for i, ts in enumerate(all_timestamps) if ts.hour == 12]
                if noon_times:
                    st.session_state.timestamp_idx = noon_times[0]
        
        with control_cols[7]:
            if st.button("⏰ 3:15", use_container_width=True):
                close_times = [i for i, ts in enumerate(all_timestamps) if ts.hour == 15 and ts.minute >= 15]
                if close_times:
                    st.session_state.timestamp_idx = close_times[0]
        
        timestamp_options = [ts.strftime('%H:%M IST') for ts in all_timestamps]
        
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            st.markdown(f"""<div class="metric-card neutral" style="padding: 15px;">
                <div class="metric-label">Start Time</div>
                <div class="metric-value" style="font-size: 1.2rem;">{timestamp_options[0]}</div>
            </div>""", unsafe_allow_html=True)
        
        with col2:
            if 'timestamp_idx' not in st.session_state:
                st.session_state.timestamp_idx = len(all_timestamps) - 1
            
            selected_timestamp_idx = st.slider(
                "🎯 Drag to navigate through intraday data points",
                min_value=0,
                max_value=len(all_timestamps) - 1,
                value=st.session_state.timestamp_idx,
                format="",
                key="time_slider"
            )
            
            st.session_state.timestamp_idx = selected_timestamp_idx
            selected_timestamp = all_timestamps[selected_timestamp_idx]
            
            progress = (selected_timestamp_idx + 1) / len(all_timestamps)
            st.progress(progress)
            
            st.info(f"📍 **{selected_timestamp.strftime('%H:%M:%S IST')}** | Point {selected_timestamp_idx + 1} of {len(all_timestamps)}")
        
        with col3:
            st.markdown(f"""<div class="metric-card neutral" style="padding: 15px;">
                <div class="metric-label">End Time</div>
                <div class="metric-value" style="font-size: 1.2rem;">{timestamp_options[-1]}</div>
            </div>""", unsafe_allow_html=True)
        
        df_selected = df[df['timestamp'] == selected_timestamp].copy()
        
        if len(df_selected) == 0:
            closest_idx = min(range(len(all_timestamps)), 
                             key=lambda i: abs((all_timestamps[i] - selected_timestamp).total_seconds()))
            df_selected = df[df['timestamp'] == all_timestamps[closest_idx]].copy()
        
        df_latest = df_selected
        spot_price = df_latest['spot_price'].iloc[0] if len(df_latest) > 0 else 0
        
        # Calculate strike range for nearest 6 strikes (±3 from ATM)
        config = SYMBOL_CONFIG.get(symbol, SYMBOL_CONFIG["NIFTY"])
        strike_interval = config["strike_interval"]
        
        strike_range = 3 * strike_interval
        df_calc = df_latest[
            (df_latest['strike'] >= spot_price - strike_range) & 
            (df_latest['strike'] <= spot_price + strike_range)
        ].copy()
        
        # Use filtered data for metrics calculation
        total_gex = df_calc['net_gex'].sum()
        total_dex = df_calc['net_dex'].sum()
        total_net = total_gex + total_dex
        total_call_oi = df_calc['call_oi'].sum()
        total_put_oi = df_calc['put_oi'].sum()
        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1
        
        # NEW: Identify gamma flip zones
        flip_zones = identify_gamma_flip_zones(df_latest, spot_price)
        
        st.markdown("### 📊 Historical Data Overview")
        
        # NEW: Info about flip zones if detected
        if len(flip_zones) > 0:
            flip_info = " | ".join([f"🔄 Flip @ ₹{z['strike']:,.0f} {z['arrow']}" for z in flip_zones[:3]])
            st.info(f"""
            📊 **Calculation Method**: Market metrics below are calculated using only the **nearest 6 strikes (±3 from ATM)** around spot price ₹{spot_price:,.2f}. 
            
            🎯 **Gamma Flip Zones Detected**: {flip_info}
            
            The arrow (↑/↓) shows the valid flip direction based on spot position relative to the flip zone.
            """)
        else:
            st.info(f"""
            📊 **Calculation Method**: Market metrics below are calculated using only the **nearest 6 strikes (±3 from ATM)** around spot price ₹{spot_price:,.2f}. 
            This focuses on strikes with actual market impact. All selected strikes are displayed in charts for comprehensive analysis.
            """)
        
        cols = st.columns(6)
        
        with cols[0]:
            st.markdown(f"""<div class="metric-card neutral">
                <div class="metric-label">Date</div>
                <div class="metric-value" style="font-size: 1.2rem;">{target_date}</div>
                <div class="metric-delta">{selected_timestamp.strftime('%H:%M:%S IST')}</div>
            </div>""", unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown(f"""<div class="metric-card neutral">
                <div class="metric-label">Spot Price</div>
                <div class="metric-value">₹{spot_price:,.2f}</div>
                <div class="metric-delta">@ Selected Time</div>
            </div>""", unsafe_allow_html=True)
        
        with cols[2]:
            gex_class = "positive" if total_gex > 0 else "negative"
            st.markdown(f"""<div class="metric-card {gex_class}">
                <div class="metric-label">Total NET GEX</div>
                <div class="metric-value {gex_class}">{total_gex:.4f}B</div>
                <div class="metric-delta">{'Suppression' if total_gex > 0 else 'Amplification'}</div>
            </div>""", unsafe_allow_html=True)
        
        with cols[3]:
            dex_class = "positive" if total_dex > 0 else "negative"
            st.markdown(f"""<div class="metric-card {dex_class}">
                <div class="metric-label">Total NET DEX</div>
                <div class="metric-value {dex_class}">{total_dex:.4f}B</div>
                <div class="metric-delta">{'Bullish' if total_dex > 0 else 'Bearish'}</div>
            </div>""", unsafe_allow_html=True)
        
        with cols[4]:
            net_class = "positive" if total_net > 0 else "negative"
            st.markdown(f"""<div class="metric-card {net_class}">
                <div class="metric-label">GEX + DEX</div>
                <div class="metric-value {net_class}">{total_net:.4f}B</div>
                <div class="metric-delta">Combined Signal</div>
            </div>""", unsafe_allow_html=True)
        
        with cols[5]:
            pcr_class = "positive" if pcr > 1 else "negative"
            st.markdown(f"""<div class="metric-card {pcr_class}">
                <div class="metric-label">Put/Call Ratio</div>
                <div class="metric-value {pcr_class}">{pcr:.2f}</div>
                <div class="metric-delta">{'Bearish' if pcr > 1.2 else 'Bullish' if pcr < 0.8 else 'Neutral'}</div>
            </div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        cols = st.columns(5)
        with cols[0]:
            gex_signal = "🟢 GEX SUPPRESSION" if total_gex > 0 else "🔴 GEX AMPLIFICATION"
            gex_badge = "bullish" if total_gex > 0 else "bearish"
            st.markdown(f'<div class="signal-badge {gex_badge}">{gex_signal}</div>', unsafe_allow_html=True)
        
        with cols[1]:
            dex_signal = "🟢 DEX BULLISH" if total_dex > 0 else "🔴 DEX BEARISH"
            dex_badge = "bullish" if total_dex > 0 else "bearish"
            st.markdown(f'<div class="signal-badge {dex_badge}">{dex_signal}</div>', unsafe_allow_html=True)
        
        with cols[2]:
            net_signal = "🟢 NET POSITIVE" if total_net > 0 else "🔴 NET NEGATIVE"
            net_badge = "bullish" if total_net > 0 else "bearish"
            st.markdown(f'<div class="signal-badge {net_badge}">{net_signal}</div>', unsafe_allow_html=True)
        
        with cols[3]:
            st.markdown(f'<div class="signal-badge volatile">📊 {len(df_latest)} Strikes</div>', unsafe_allow_html=True)
        
        with cols[4]:
            # NEW: Show flip zones badge
            if len(flip_zones) > 0:
                st.markdown(f'<div class="signal-badge volatile">🔄 {len(flip_zones)} Flip Zones</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # TABS - WITH CACHE MANAGEMENT
        tabs = st.tabs(["📊 NET GEX", "📊 NET DEX", "🎯 GEX", "📊 DEX", "⚡ NET GEX+DEX", 
                        "🎪 Hedge Pressure", "🌊 NET GEX Flow", "🌊 NET DEX Flow", 
                        "🌊 VANNA", "⏰ CHARM", "📈 Intraday Timeline", "📋 OI & Data", "💾 Cache Manager"])
        
        # Tab 0: NET GEX with Flip Zones
        with tabs[0]:
            st.markdown("### 📊 NET Gamma Exposure (NET GEX) with Flip Zones")
            st.markdown(f"*Calculated using nearest 6 strikes (±3 from ATM at ₹{spot_price:,.0f})*")
            st.plotly_chart(create_separate_gex_chart(df_latest, spot_price), use_container_width=True, key="net_gex_chart")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total NET GEX (Nearest 6 Strikes)", f"{total_gex:.4f}B")
            with col2:
                gex_status = "Volatility Suppression (Range-Bound)" if total_gex > 0 else "Volatility Amplification (Trending)"
                st.info(f"📌 Market Status: {gex_status}")
            
            # NEW: Show flip zone details
            if len(flip_zones) > 0:
                st.markdown("#### 🔄 Gamma Flip Zones Detected")
                for zone in flip_zones:
                    st.markdown(f"""
                    - **Flip @ ₹{zone['strike']:,.0f}** {zone['arrow']} | Type: {zone['flip_type']} | 
                    Valid Direction: {'Moving Up' if zone['direction'] == 'upward' else 'Moving Down'}
                    """)
        
        # Tab 1: NET DEX (ORIGINAL)
        with tabs[1]:
            st.markdown("### 📊 NET Delta Exposure (NET DEX)")
            st.markdown(f"*Calculated using nearest 6 strikes (±3 from ATM at ₹{spot_price:,.0f})*")
            st.plotly_chart(create_separate_dex_chart(df_latest, spot_price), use_container_width=True, key="net_dex_chart")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total NET DEX (Nearest 6 Strikes)", f"{total_dex:.4f}B")
            with col2:
                dex_status = "Bullish Positioning" if total_dex > 0 else "Bearish Positioning"
                st.info(f"📌 Market Direction: {dex_status}")
        
        # Tab 2: GEX with Flip Zones
        with tabs[2]:
            st.markdown("### 🎯 Gamma Exposure (GEX) Analysis with Flip Zones")
            st.plotly_chart(create_separate_gex_chart(df_latest, spot_price), use_container_width=True, key="gex_chart")
            
            col1, col2 = st.columns(2)
            with col1:
                positive_gex = df_latest[df_latest['net_gex'] > 0]['net_gex'].sum()
                st.metric("Positive GEX", f"{positive_gex:.4f}B")
            with col2:
                negative_gex = df_latest[df_latest['net_gex'] < 0]['net_gex'].sum()
                st.metric("Negative GEX", f"{negative_gex:.4f}B")
        
        # Tab 3: DEX (ORIGINAL)
        with tabs[3]:
            st.markdown("### 📊 Delta Exposure (DEX) Analysis")
            st.plotly_chart(create_separate_dex_chart(df_latest, spot_price), use_container_width=True, key="dex_chart")
            
            col1, col2 = st.columns(2)
            with col1:
                positive_dex = df_latest[df_latest['net_dex'] > 0]['net_dex'].sum()
                st.metric("Positive DEX", f"{positive_dex:.4f}B")
            with col2:
                negative_dex = df_latest[df_latest['net_dex'] < 0]['net_dex'].sum()
                st.metric("Negative DEX", f"{negative_dex:.4f}B")
        
        # Tab 4: NET GEX+DEX with Flip Zones
        with tabs[4]:
            st.markdown("### ⚡ Combined NET GEX + DEX Analysis with Flip Zones")
            st.plotly_chart(create_net_gex_dex_chart(df_latest, spot_price), use_container_width=True, key="net_gex_dex_chart")
        
        # Tab 5: Hedge Pressure with Flip Zones
        with tabs[5]:
            st.markdown("### 🎪 Hedging Pressure Distribution with Flip Zones")
            st.plotly_chart(create_hedging_pressure_chart(df_latest, spot_price), use_container_width=True, key="hedge_pressure_chart")
        
        # Tab 6-11: ORIGINAL TABS
        with tabs[6]:
            st.markdown("### 🌊 NET GEX Flow Analysis")
            st.plotly_chart(create_net_gex_flow_chart(df_latest, spot_price), use_container_width=True, key="net_gex_flow_chart")
            
            total_gex_inflow = df_latest[df_latest['net_gex_flow'] > 0]['net_gex_flow'].sum()
            total_gex_outflow = df_latest[df_latest['net_gex_flow'] < 0]['net_gex_flow'].sum()
            net_gex_flow = total_gex_inflow + total_gex_outflow
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("GEX Inflow", f"{total_gex_inflow:.4f}B")
            with col2:
                st.metric("GEX Outflow", f"{total_gex_outflow:.4f}B")
            with col3:
                st.metric("NET GEX Flow", f"{net_gex_flow:.4f}B")
        
        with tabs[7]:
            st.markdown("### 🌊 NET DEX Flow Analysis")
            st.plotly_chart(create_net_dex_flow_chart(df_latest, spot_price), use_container_width=True, key="net_dex_flow_chart")
            
            total_dex_inflow = df_latest[df_latest['net_dex_flow'] > 0]['net_dex_flow'].sum()
            total_dex_outflow = df_latest[df_latest['net_dex_flow'] < 0]['net_dex_flow'].sum()
            net_dex_flow = total_dex_inflow + total_dex_outflow
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("DEX Inflow", f"{total_dex_inflow:.4f}B")
            with col2:
                st.metric("DEX Outflow", f"{total_dex_outflow:.4f}B")
            with col3:
                st.metric("NET DEX Flow", f"{net_dex_flow:.4f}B")
        
        with tabs[8]:
            st.markdown("### 🌊 VANNA Exposure")
            st.plotly_chart(create_vanna_exposure_chart(df_latest, spot_price), use_container_width=True, key="vanna_chart")
            
            total_call_vanna = df_latest['call_vanna'].sum()
            total_put_vanna = df_latest['put_vanna'].sum()
            net_vanna = df_latest['net_vanna'].sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Call VANNA", f"{total_call_vanna:.4f}B")
            with col2:
                st.metric("Put VANNA", f"{total_put_vanna:.4f}B")
            with col3:
                st.metric("Net VANNA", f"{net_vanna:.4f}B")
        
        with tabs[9]:
            st.markdown("### ⏰ CHARM Exposure")
            st.plotly_chart(create_charm_exposure_chart(df_latest, spot_price), use_container_width=True, key="charm_chart")
            
            total_call_charm = df_latest['call_charm'].sum()
            total_put_charm = df_latest['put_charm'].sum()
            net_charm = df_latest['net_charm'].sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Call CHARM", f"{total_call_charm:.4f}B")
            with col2:
                st.metric("Put CHARM", f"{total_put_charm:.4f}B")
            with col3:
                st.metric("Net CHARM", f"{net_charm:.4f}B")
        
        with tabs[10]:
            st.markdown("### 📈 Intraday GEX/DEX Evolution")
            st.plotly_chart(create_intraday_timeline(df, selected_timestamp), use_container_width=True, key="intraday_timeline_chart")
        
        with tabs[11]:
            st.markdown("### 📋 Open Interest Distribution")
            st.plotly_chart(create_oi_distribution(df_latest, spot_price), use_container_width=True, key="oi_distribution_chart")
            
            st.markdown("### 📊 Complete Data Table")
            display_df = df_latest[['strike', 'call_oi', 'put_oi', 'total_volume', 'net_gex', 'net_dex']].copy()
            display_df['net_gex'] = display_df['net_gex'].apply(lambda x: f"{x:.4f}B")
            display_df['net_dex'] = display_df['net_dex'].apply(lambda x: f"{x:.4f}B")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)
            
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Full Historical Data (CSV)",
                data=csv,
                file_name=f"NYZTrade_{symbol}_{target_date}.csv",
                mime="text/csv"
            )
        
        # Tab 12: Cache Management
        with tabs[12]:
            st.markdown("### 💾 Cache Management")
            
            cache_stats = cache_manager.get_cache_stats()
            
            # Cache Statistics
            st.markdown("#### 📊 Cache Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Entries", cache_stats['total_entries'])
            with col2:
                st.metric("Cache Size", f"{cache_stats['total_size_mb']:.2f} MB")
            with col3:
                st.metric("Unique Dates", cache_stats['unique_dates'])
            with col4:
                st.metric("Symbols", len(cache_stats['symbols']))
            
            if cache_stats['total_entries'] > 0:
                st.info(f"📅 Date Range: {cache_stats['oldest_date']} to {cache_stats['newest_date']}")
            
            st.markdown("---")
            
            # Batch Download Section
            st.markdown("#### 📥 Batch Download Historical Data")
            st.info("💡 Download multiple dates at once to build your 5-year historical cache")
            
            batch_col1, batch_col2 = st.columns(2)
            
            with batch_col1:
                batch_symbol = st.selectbox(
                    "Symbol for Batch Download",
                    list(DHAN_SECURITY_IDS.keys()),
                    key="batch_symbol"
                )
                
                batch_start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365),  # 1 year ago
                    max_value=datetime.now(),
                    min_value=datetime.now() - timedelta(days=365*5),  # 5 years ago
                    key="batch_start"
                )
                
                batch_end_date = st.date_input(
                    "End Date",
                    value=datetime.now() - timedelta(days=7),  # 1 week ago
                    max_value=datetime.now(),
                    min_value=batch_start_date,
                    key="batch_end"
                )
            
            with batch_col2:
                batch_strikes = st.multiselect(
                    "Select Strikes",
                    ["ATM", "ATM+1", "ATM-1", "ATM+2", "ATM-2", "ATM+3", "ATM-3"],
                    default=["ATM", "ATM+1", "ATM-1", "ATM+2", "ATM-2"],
                    key="batch_strikes"
                )
                
                batch_interval = st.selectbox(
                    "Interval",
                    ["60", "15", "5"],
                    format_func=lambda x: "1 hour" if x == "60" else "15 minutes" if x == "15" else "5 minutes",
                    key="batch_interval"
                )
                
                batch_expiry = st.selectbox(
                    "Expiry Type",
                    ["Weekly", "Monthly"],
                    key="batch_expiry"
                )
            
            # Calculate number of trading days
            total_days = (batch_end_date - batch_start_date).days + 1
            estimated_trading_days = int(total_days * 5 / 7)  # Rough estimate
            
            st.caption(f"📅 Will attempt to download ~{estimated_trading_days} trading days")
            st.warning("⚠️ Batch downloads can take 10-30 minutes depending on date range and API rate limits")
            
            if st.button("🚀 Start Batch Download", type="primary", use_container_width=True):
                if not batch_strikes:
                    st.error("Please select at least one strike")
                else:
                    fetcher = DhanHistoricalFetcher(DhanConfig())
                    expiry_flag = "WEEK" if batch_expiry == "Weekly" else "MONTH"
                    
                    stats = cache_manager.batch_download_dates(
                        symbol=batch_symbol,
                        start_date=batch_start_date.strftime('%Y-%m-%d'),
                        end_date=batch_end_date.strftime('%Y-%m-%d'),
                        fetcher=fetcher,
                        strikes=batch_strikes,
                        interval=batch_interval,
                        expiry_code=1,
                        expiry_flag=expiry_flag
                    )
                    
                    st.success(f"✅ Batch download complete!")
                    st.info(f"""
                    **Results:**
                    - Successful: {stats['successful']}
                    - Failed: {stats['failed']}
                    - Already Cached: {stats['skipped_cached']}
                    """)
            
            st.markdown("---")
            
            # Cached Dates Browser
            st.markdown("#### 📅 Browse Cached Dates")
            
            browse_symbol = st.selectbox(
                "Filter by Symbol",
                ["All"] + list(DHAN_SECURITY_IDS.keys()),
                key="browse_symbol"
            )
            
            symbol_filter = None if browse_symbol == "All" else browse_symbol
            cached_dates = cache_manager.get_cached_dates(symbol=symbol_filter)
            
            if cached_dates:
                st.success(f"Found {len(cached_dates)} cached dates")
                
                # Show dates in a grid
                dates_per_row = 5
                for i in range(0, len(cached_dates[:50]), dates_per_row):  # Show first 50
                    cols = st.columns(dates_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(cached_dates[:50]):
                            col.caption(cached_dates[i + j])
                
                if len(cached_dates) > 50:
                    st.caption(f"... and {len(cached_dates) - 50} more dates")
            else:
                st.info("No cached data yet. Use batch download above to start building your cache.")
            
            st.markdown("---")
            
            # Cache Management Actions
            st.markdown("#### 🗑️ Cache Management")
            
            mgmt_col1, mgmt_col2 = st.columns(2)
            
            with mgmt_col1:
                st.markdown("**Clear Cache**")
                clear_symbol = st.selectbox(
                    "Clear Symbol",
                    ["All"] + list(DHAN_SECURITY_IDS.keys()),
                    key="clear_symbol"
                )
                
                if st.button("🗑️ Clear Selected Cache", type="secondary"):
                    symbol_to_clear = None if clear_symbol == "All" else clear_symbol
                    removed = cache_manager.clear_cache(symbol=symbol_to_clear)
                    st.success(f"Removed {removed} cache entries")
                    st.rerun()
            
            with mgmt_col2:
                st.markdown("**Export Cache**")
                st.info("💡 Export cached data to CSV for backup or external analysis")
                
                if cache_stats['total_entries'] > 0:
                    if st.button("📤 Export All to CSV"):
                        st.info("Batch export feature - coming soon!")
                else:
                    st.caption("No cached data to export")
    
    else:
        st.info("""
        👋 **Welcome to NYZTrade Historical GEX/DEX Dashboard with 5-Year Cache!**
        
        **New Features:**
        - 🔄 **Gamma Flip Zones** - Identifies critical GEX zero-crossing levels with directional arrows
        - 💾 **5-Year Data Cache** - Store and access historical data locally for instant analysis
        - 📥 **Batch Download** - Download months/years of data at once
        
        **Getting Started:**
        1. Go to **💾 Cache Manager** tab to download historical data
        2. Use **Batch Download** to build your 5-year cache
        3. Once cached, data loads instantly (no API calls needed!)
        
        **Gamma Flip Zones Explained:**
        - **What**: Strike levels where GEX changes from positive to negative (or vice versa)
        - **Why**: Critical levels that affect dealer hedging behavior
        - **Arrows**: Show the valid flip direction:
          - ↑ (Up Arrow): Flip valid when price moves UP through this level
          - ↓ (Down Arrow): Flip valid when price moves DOWN through this level
        - **Colors**: 
          - 🟢 Green: Flip leads to suppression (stabilization)
          - 🔴 Red: Flip leads to amplification (acceleration)
        
        **How to use:**
        1. Select index and date (or batch download dates first)
        2. Choose strikes (up to ±10)
        3. Click "Fetch Historical Data" (loads from cache if available)
        4. Navigate through 13 comprehensive tabs
        
        **💡 Pro Tip:** 
        - Build your cache first using Batch Download for instant access to years of data
        - Watch for price approaching gamma flip zones - critical decision points for dealers!
        """)
    
    st.markdown("---")
    st.markdown(f"""<div style="text-align: center; padding: 20px; color: #64748b;">
        <p style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
        NYZTrade Historical GEX/DEX Dashboard | 5-Year Cache System | Gamma Flip Zones<br>
        13 Analysis Tabs | Extended Strikes (±10) | Batch Download Support</p>
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
