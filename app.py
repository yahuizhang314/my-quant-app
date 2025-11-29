import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import calendar
from datetime import datetime

# --- 1. 基础配置 ---
st.set_page_config(
    page_title="Investment Notes | MyQuant Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 视觉升级 (Rational Luxury Design System) ---
def apply_design_system():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Noto+Serif+SC:wght@400;700&family=Playfair+Display:ital,wght@0,400;0,600;1,400&display=swap');

            :root {
                --primary-color: #1a1a1a;
                --accent-color: #002FA7; /* Klein Blue */
                --bg-color: #ffffff;
                --secondary-bg: #f8f9fa;
                --text-color: #1a1a1a;
                --subtext-color: #666666;
                --border-color: #e0e0e0;
            }

            /* 全局字体 */
            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
                color: var(--text-color);
                background-color: var(--bg-color);
            }

            /* 标题字体：衬线体营造高级感 */
            h1, h2, h3 {
                font-family: 'Playfair Display', 'Noto Serif SC', serif !important;
                font-weight: 700 !important;
                color: var(--primary-color) !important;
                letter-spacing: -0.5px;
            }
            
            h1 { font-size: 2.5rem !important; margin-bottom: 1rem !important; }
            h2 { font-size: 1.5rem !important; border-bottom: 2px solid var(--accent-color); padding-bottom: 10px; display: inline-block; }
            h3 { font-size: 1.2rem !important; margin-top: 20px !important; }

            /* 侧边栏 */
            [data-testid="stSidebar"] {
                background-color: var(--secondary-bg);
                border-right: 1px solid var(--border-color);
            }
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
                font-family: 'Inter', sans-serif !important;
                text-transform: uppercase;
                font-size: 0.8rem !important;
                letter-spacing: 1px;
                color: var(--subtext-color) !important;
                border-bottom: none;
            }

            /* 按钮设计：胶囊形状 */
            div.stButton > button {
                border-radius: 50px;
                background-color: transparent;
                color: var(--primary-color);
                border: 1px solid var(--primary-color);
                font-family: 'Inter', sans-serif;
                text-transform: uppercase;
                letter-spacing: 1px;
                font-size: 0.8rem;
                padding: 0.5rem 1.5rem;
                transition: all 0.3s ease;
            }
            div.stButton > button:hover {
                background-color: var(--accent-color);
                border-color: var(--accent-color);
                color: white;
            }

            /* 指标卡片 (Metrics) */
            [data-testid="stMetric"] {
                background-color: white;
                border: 1px solid var(--border-color);
                padding: 15px;
                border-radius: 0px;
                box-shadow: none;
                transition: 0.3s;
            }
            [data-testid="stMetric"]:hover {
                border-color: var(--accent-color);
                transform: translateY(-2px);
            }
            [data-testid="stMetricLabel"] {
                font-family: 'Inter', sans-serif;
                font-size: 0.8rem !important;
                text-transform: uppercase;
                color: var(--subtext-color) !important;
            }
            [data-testid="stMetricValue"] {
                font-family: 'Playfair Display', serif;
                font-size: 1.8rem !important;
                color: var(--accent-color) !important;
            }

            /* Expander */
            .streamlit-expanderHeader {
                font-family: 'Inter', sans-serif;
                font-size: 0.9rem;
                background-color: var(--secondary-bg);
                border-radius: 4px;
            }

            /* Tabs */
            .stTabs [data-baseweb="tab-list"] { gap: 20px; }
            .stTabs [data-baseweb="tab"] {
                font-family: 'Playfair Display', serif;
                font-size: 1.1rem;
                color: var(--subtext-color);
                padding-bottom: 5px;
            }
            .stTabs [aria-selected="true"] {
                color: var(--accent-color) !important;
                font-weight: bold;
                border-bottom: 2px solid var(--accent-color) !important;
            }
            
            /* 数据表格 */
            [data-testid="stDataFrame"] { font-family: 'Inter', sans-serif; font-size: 0.9rem; }
            
            /* 分割线 */
            hr { border-color: var(--border-color); margin: 30px 0; }
        </style>
    """, unsafe_allow_html=True)

apply_design_system()

# --- 3. 核心计算函数库 ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig', thousands=',')
        df.columns = df.columns.str.strip().str.lower()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        for col in ['close', 'open', 'high', 'low', 'volume']:
             df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['close'], inplace=True)
        return df
    except Exception as e:
        return None

# --- 技术指标计算 ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = (dif - dea) * 2
    return dif, dea, macd_hist

def calculate_bollinger(df, window=20, std_dev=2):
    mid = df['close'].rolling(window=window).mean()
    std = df['close'].rolling(window=window).std()
    upper = mid + (std * std_dev)
    lower = mid - (std * std_dev)
    return upper, mid, lower

def calculate_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=window).mean()

def calculate_kdj(df, period=9, k_period=3, d_period=3):
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(alpha=1/k_period, adjust=False).mean()
    d = k.ewm(alpha=1/d_period, adjust=False).mean()
    return k, d

def calculate_cci(df, window=14):
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window).mean()
    mad = (tp - sma).abs().rolling(window).mean()
    cci = (tp - sma) / (0.015 * mad)
    return cci

# --- 策略解释文案 ---
STRATEGY_INFO = {
    "双均线 (Dual MA)": """
    **📝 核心逻辑：** 趋势跟踪策略。
    * **买入信号：** 当【短期均线】上穿【长期均线】时（金叉），认为趋势向上，买入。
    * **卖出信号：** 当【短期均线】下穿【长期均线】时（死叉），认为趋势结束，卖出。
    * **适用场景：** 单边大牛市或大熊市。震荡市容易反复止损。
    """,
    "MACD 趋势 (MACD Trend)": """
    **📝 核心逻辑：** 经典的趋势指标策略。
    * **买入信号：** 当 DIF 线（快线）上穿 DEA 线（慢线）时，形成 MACD 金叉，买入。
    * **卖出信号：** 当 DIF 线下穿 DEA 线时，形成 MACD 死叉，卖出。
    * **特点：** 相比普通均线，MACD 对价格变化的反应稍快。
    """,
    "动量突破 (Momentum)": """
    **📝 核心逻辑：** 强者恒强。
    * **买入信号：** 当过去 N 天的涨幅超过设定的阈值（例如 2%）时，全仓追入。
    * **卖出信号：** 当动量消失（今日价格低于 N 天前）时卖出。
    * **适用场景：** 大牛市主升浪。
    """,
    "RSI 反转 (RSI Reversion)": """
    **📝 核心逻辑：** 均值回归。
    * **买入信号：** RSI < 买入阈值（通常30），代表市场超卖，博反弹。
    * **卖出信号：** RSI > 卖出阈值（通常70），代表市场超买，落袋为安。
    * **适用场景：** 震荡市、箱体整理行情。
    """,
    "布林带回归 (Bollinger)": """
    **📝 核心逻辑：** 利用统计学标准差捕捉价格异常。
    * **买入信号：** 价格跌破【布林下轨】，认为跌过头了，进场抄底。
    * **卖出信号：** 价格突破【布林上轨】，认为涨过头了，卖出。
    * **适用场景：** 震荡行情。
    """,
    "海龟交易 (Donchian)": """
    **📝 核心逻辑：** 通道突破 (Trend)。
    * **买入：** 收盘价突破过去 N 天的最高价 (创近期新高)。
    * **卖出：** 收盘价跌破过去 M 天的最低价。
    * **特点：** 著名的海龟法则核心，抓住大趋势，不怕震荡止损。
    """,
    "KDJ 随机 (KDJ Stochastic)": """
    **📝 核心逻辑：** 经典反转。
    * **买入：** K < 20 且 K线上穿D线 (低位金叉)。
    * **卖出：** K > 80 且 K线下穿D线 (高位死叉)。
    * **特点：** A股常用指标，对短线转折非常敏感。
    """,
    "ATR 波动突破 (ATR Breakout)": """
    **📝 核心逻辑：** 波动率趋势。
    * **买入：** 今日收盘 > 昨日收盘 + (系数 * ATR)。意味着价格伴随波动率大幅向上突破。
    * **卖出：** 跌破均线离场。
    * **特点：** 过滤掉窄幅波动的假突破。
    """,
    "量价齐升 (Vol & Price)": """
    **📝 核心逻辑：** 量价配合。
    * **买入：** 收盘价 > N日均线 **且** 成交量 > N日均量 (放量上涨)。
    * **卖出：** 收盘价 < N日均线。
    * **特点：** 确认资金进场，信号更可靠。
    """,
    "CCI 顺势 (CCI Trend)": """
    **📝 核心逻辑：** 极端行情。
    * **买入：** CCI > 100 (进入强势拉升区)。
    * **卖出：** CCI < 100 (脱离强势区)。
    * **特点：** 专门捕捉暴涨暴跌的单边行情，不做震荡。
    """
}

# --- 4. 策略引擎 ---
def run_strategy_engine(df_origin, strategy_type, params, initial_cash):
    df = df_origin.copy()
    
    # 预计算指标
    df['MA_S'] = df['close'].rolling(params.get('ma_short', 5)).mean()
    df['MA_L'] = df['close'].rolling(params.get('ma_long', 20)).mean()
    df['Vol_MA'] = df['volume'].rolling(params.get('vol_ma', 20)).mean()
    mom_days = params.get('mom_days', 10)
    df['Momentum'] = df['close'].pct_change(periods=mom_days)
    df['RSI'] = calculate_rsi(df['close'], params.get('rsi_window', 14))
    df['DIF'], df['DEA'], _ = calculate_macd(df, params.get('macd_fast',12), params.get('macd_slow',26), params.get('macd_sig',9))
    df['BB_Up'], _, df['BB_Low'] = calculate_bollinger(df, params.get('bb_window',20), params.get('bb_std',2))
    don_up = params.get('don_up', 20); don_down = params.get('don_down', 10)
    df['Don_High'] = df['high'].rolling(don_up).max().shift(1)
    df['Don_Low'] = df['low'].rolling(don_down).min().shift(1)
    df['ATR'] = calculate_atr(df, 14)
    df['K'], df['D'] = calculate_kdj(df, 9, 3, 3)
    df['CCI'] = calculate_cci(df, 14)

    cash = initial_cash
    shares = 0
    total_assets = []
    trades = [] # 记录每一笔交易详情
    
    entry_price = 0
    highest_price = 0 
    
    closes = df['close'].values; vols = df['volume'].values
    dates = df.index
    n = len(df)
    
    # Numpy 加速
    ind_ma_s = df['MA_S'].values; ind_ma_l = df['MA_L'].values
    ind_rsi = df['RSI'].values; ind_dif = df['DIF'].values; ind_dea = df['DEA'].values
    ind_bb_low = df['BB_Low'].values; ind_bb_up = df['BB_Up'].values
    ind_mom = df['Momentum'].values; ind_don_h = df['Don_High'].values; ind_don_l = df['Don_Low'].values
    ind_k = df['K'].values; ind_d = df['D'].values; ind_atr = df['ATR'].values
    ind_vol_ma = df['Vol_MA'].values; ind_cci = df['CCI'].values
    
    sl_pct = params.get('sl_pct', 0.05)
    tp_pct = params.get('tp_pct', 0.20)
    use_trailing = params.get('use_trailing', False)
    
    for i in range(n):
        price = closes[i]
        current_date = dates[i]
        
        if shares == 0:
            buy_signal = False
            # --- 10 大策略逻辑 ---
            if strategy_type == "双均线 (Dual MA)":
                if ind_ma_s[i] > ind_ma_l[i] and not np.isnan(ind_ma_l[i]): buy_signal = True
            elif strategy_type == "RSI 反转 (RSI Reversion)":
                if ind_rsi[i] < params['rsi_buy'] and not np.isnan(ind_rsi[i]): buy_signal = True
            elif strategy_type == "MACD 趋势 (MACD Trend)":
                if i>0 and ind_dif[i] > ind_dea[i] and ind_dif[i-1] <= ind_dea[i-1] and not np.isnan(ind_dea[i]): buy_signal = True
            elif strategy_type == "布林带回归 (Bollinger)":
                if price < ind_bb_low[i] and not np.isnan(ind_bb_low[i]): buy_signal = True
            elif strategy_type == "动量突破 (Momentum)":
                if ind_mom[i] > params['mom_thres'] and not np.isnan(ind_mom[i]): buy_signal = True
            elif strategy_type == "海龟交易 (Donchian)":
                if price > ind_don_h[i] and not np.isnan(ind_don_h[i]): buy_signal = True
            elif strategy_type == "KDJ 随机 (KDJ Stochastic)":
                if ind_k[i] < 20 and ind_k[i] > ind_d[i] and ind_k[i-1] <= ind_d[i-1] and not np.isnan(ind_k[i]): buy_signal = True
            elif strategy_type == "ATR 波动突破 (ATR Breakout)":
                if i>0 and not np.isnan(ind_atr[i]) and price > closes[i-1] + params['atr_k'] * ind_atr[i]: buy_signal = True
            elif strategy_type == "量价齐升 (Vol & Price)":
                if not np.isnan(ind_ma_l[i]) and not np.isnan(ind_vol_ma[i]):
                    if price > ind_ma_l[i] and vols[i] > ind_vol_ma[i]: buy_signal = True
            elif strategy_type == "CCI 顺势 (CCI Trend)":
                if ind_cci[i] > 100 and not np.isnan(ind_cci[i]): buy_signal = True

            if buy_signal:
                shares = int(cash / price)
                cash -= shares * price
                entry_price = price
                highest_price = price
                
                # 记录买入 (Dict)
                current_val = cash + shares * price
                trades.append({
                    "date": current_date,
                    "type": "Buy",
                    "price": price,
                    "reason": "Signal",
                    "pnl": 0,
                    "total_asset": current_val
                })

        else:
            sell_type = None
            if price > highest_price: highest_price = price
            
            # 风控
            if use_trailing:
                drawdown = (highest_price - price) / highest_price
                if drawdown >= sl_pct: sell_type = "移动止损 🛑"
            else:
                pct_change = (price - entry_price) / entry_price
                if pct_change <= -sl_pct: sell_type = "固定止损 🛑"
            
            # 止盈
            total_gain = (price - entry_price) / entry_price
            if not sell_type and total_gain >= tp_pct: sell_type = "止盈 💰"
            
            # 策略平仓
            if not sell_type:
                if strategy_type == "双均线 (Dual MA)" and ind_ma_s[i] < ind_ma_l[i]: sell_type = "死叉离场"
                elif strategy_type == "RSI 反转 (RSI Reversion)" and ind_rsi[i] > params['rsi_sell']: sell_type = "RSI高位离场"
                elif strategy_type == "MACD 趋势 (MACD Trend)" and ind_dif[i] < ind_dea[i]: sell_type = "MACD死叉"
                elif strategy_type == "布林带回归 (Bollinger)" and price > ind_bb_up[i]: sell_type = "触顶回调"
                elif strategy_type == "动量突破 (Momentum)" and ind_mom[i] < 0: sell_type = "动量消失"
                elif strategy_type == "海龟交易 (Donchian)" and price < ind_don_l[i]: sell_type = "跌破下轨"
                elif strategy_type == "KDJ 随机 (KDJ Stochastic)" and (ind_k[i] > 80 and ind_k[i] < ind_d[i]): sell_type = "KDJ死叉"
                elif strategy_type == "ATR 波动突破 (ATR Breakout)" and price < ind_ma_l[i]: sell_type = "跌破均线"
                elif strategy_type == "量价齐升 (Vol & Price)" and price < ind_ma_l[i]: sell_type = "趋势破位"
                elif strategy_type == "CCI 顺势 (CCI Trend)" and ind_cci[i] < 100: sell_type = "CCI转弱"
            
            if sell_type:
                cash += shares * price
                shares = 0
                
                # 记录卖出 (Dict)
                current_val = cash
                trades.append({
                    "date": current_date,
                    "type": "Sell",
                    "price": price,
                    "reason": sell_type,
                    "pnl": (price - entry_price)/entry_price,
                    "total_asset": current_val
                })
        
        total_assets.append(cash + shares * price)
    
    df['total_asset'] = total_assets
    first_price = df['close'].iloc[0]
    df['benchmark_asset'] = initial_cash * (df['close'] / first_price)
    
    final_asset = total_assets[-1]
    final_benchmark = df['benchmark_asset'].iloc[-1]
    
    ret = (final_asset - initial_cash) / initial_cash
    bench_ret = (final_benchmark - initial_cash) / initial_cash
    
    return final_asset, final_benchmark, ret, bench_ret, trades, df

# --- 5. 侧边栏 ---
with st.sidebar:
    st.header("🎛 策略控制台")
    current_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not current_files: st.error("请放入 CSV"); st.stop()
    selected_file = st.selectbox("1. 选择标的", current_files)
    raw_df_full = load_data(selected_file)
    
    if raw_df_full is not None:
        # V15 时间选择功能
        st.caption("📅 回测时间段")
        start_d = st.date_input("开始", raw_df_full.index.min(), min_value=raw_df_full.index.min(), max_value=raw_df_full.index.max())
        end_d = st.date_input("结束", raw_df_full.index.max(), min_value=raw_df_full.index.min(), max_value=raw_df_full.index.max())
        raw_df = raw_df_full.loc[str(start_d):str(end_d)]
        if len(raw_df) < 20: st.warning("时间太短"); st.stop()
    else: st.stop()

    st.divider()
    st.subheader("2. 策略模型")
    strategies = [
        "双均线 (Dual MA)", "MACD 趋势 (MACD Trend)", "动量突破 (Momentum)", 
        "RSI 反转 (RSI Reversion)", "布林带回归 (Bollinger)",
        "海龟交易 (Donchian)", "KDJ 随机 (KDJ Stochastic)", 
        "ATR 波动突破 (ATR Breakout)", "量价齐升 (Vol & Price)", "CCI 顺势 (CCI Trend)"
    ]
    strategy_mode = st.selectbox("选择基准策略", strategies)
    params = {}
    initial_cash = st.number_input("初始资金", value=1000000, step=10000)
    
    with st.expander("🛠 策略参数微调", expanded=True):
        if strategy_mode == "双均线 (Dual MA)":
            params['ma_short'] = st.slider("短期均线", 2, 60, 5)
            params['ma_long'] = st.slider("长期均线", 10, 120, 20)
        elif strategy_mode == "MACD 趋势 (MACD Trend)":
            params['macd_fast'] = st.number_input("快线", 12); params['macd_slow'] = st.number_input("慢线", 26); params['macd_sig'] = st.number_input("信号", 9)
        elif strategy_mode == "动量突破 (Momentum)":
            params['mom_days'] = st.slider("动量周期", 5, 60, 20); params['mom_thres'] = st.slider("阈值%", 0.0, 10.0, 2.0)/100
        elif strategy_mode == "RSI 反转 (RSI Reversion)":
            params['rsi_window'] = st.slider("RSI周期", 6, 24, 14); params['rsi_buy'] = st.slider("买入", 10, 40, 30); params['rsi_sell'] = st.slider("卖出", 60, 90, 70)
        elif strategy_mode == "布林带回归 (Bollinger)":
            params['bb_window'] = st.slider("周期", 10, 60, 20); params['bb_std'] = st.slider("Std", 1.0, 3.0, 2.0)
        # 新增
        elif strategy_mode == "海龟交易 (Donchian)":
            params['don_up'] = st.slider("突破(买)", 10, 60, 20); params['don_down'] = st.slider("跌破(卖)", 5, 30, 10)
        elif strategy_mode == "KDJ 随机 (KDJ Stochastic)":
            st.caption("默认(9,3,3)")
        elif strategy_mode == "ATR 波动突破 (ATR Breakout)":
            params['atr_k'] = st.slider("ATR倍数", 0.5, 3.0, 1.0)
            params['ma_long'] = st.slider("离场均线", 10, 60, 20)
        elif strategy_mode == "量价齐升 (Vol & Price)":
            params['ma_long'] = st.slider("均线", 10, 60, 20); params['vol_ma'] = st.slider("均量", 10, 60, 20)
        elif strategy_mode == "CCI 顺势 (CCI Trend)":
            st.caption("CCI>100 买入, <100 卖出")

    st.markdown("---")
    st.subheader("3. 风险管理")
    with st.expander("📘 风控机制说明", expanded=False):
        st.markdown("**固定止损**：成本价下跌N%。\n**移动止损**：最高价回撤N%。")
        
    sl_mode = st.radio("止损模式", ["固定止损", "移动止损"], horizontal=True)
    params['use_trailing'] = True if sl_mode == "移动止损" else False
    c1, c2 = st.columns(2)
    params['sl_pct'] = c1.slider("止损%", 1.0, 20.0, 5.0)/100
    params['tp_pct'] = c2.slider("止盈%", 5.0, 100.0, 20.0)/100

# --- 6. 主界面 ---
if raw_df is not None:
    st.title(f"量化回测小站 | {selected_file}")
    st.caption("Hey there")
    
    tab1, tab2, tab3 = st.tabs(["🔍 市场体检", "⚔️ 策略回测", "🤖 智能优化"])
    
    # === Tab 1: 市场体检 (Strict V12 Design) ===
    with tab1:
        st.subheader("1. 基础概况")
        roll_max = raw_df['close'].cummax()
        max_dd = (raw_df['close'] / roll_max - 1.0).min()
        k1, k2, k3, k4 = st.columns([1.5,1,1,1])
        k1.metric("数据区间", f"{start_d} ~ {end_d}", delta_color="off")
        k2.metric("最新收盘", raw_df['close'].iloc[-1])
        k3.metric("区间涨幅", f"{((raw_df['close'].iloc[-1]/raw_df['close'].iloc[0])-1)*100:.2f}%")
        k4.metric("最大回撤", f"{max_dd*100:.2f}%")
        
        st.divider()
        st.subheader("📈 历史价格走势 (全区间)")
        st.line_chart(raw_df['close'])
        
        with st.expander("📊 查看基础统计数据 (Basic Statistics)", expanded=True):
            stats = raw_df[['open', 'high', 'low', 'close', 'volume']].describe().T
            st.dataframe(stats.style.format("{:.2f}"))
            daily_ret = raw_df['close'].pct_change()
            annual_vol = daily_ret.std() * np.sqrt(252) * 100
            st.caption(f"💡 补充指标：该标的年化波动率约为 **{annual_vol:.2f}%**")

        st.divider()
        st.subheader("📅 周期/日历效应 (Seasonality)")
        col_week, col_month = st.columns(2)
        with col_week:
            st.markdown("**周度效应**")
            df_cal = raw_df.copy(); df_cal['pct']=df_cal['close'].pct_change()*100; df_cal['wd']=df_cal.index.day_name(); df_cal['wi']=df_cal.index.dayofweek
            st.plotly_chart(px.bar(df_cal.groupby(['wi','wd'])['pct'].mean().reset_index().sort_values('wi'), x='wd', y='pct', title="周效应", color='pct', color_continuous_scale='RdBu_r'), use_container_width=True)
        with col_month:
            st.markdown("**月度效应**")
            df_cal['m']=df_cal.index.month; df_cal['mn']=df_cal.index.strftime('%b')
            mon = df_cal.groupby('m')['pct'].mean().reset_index(); mon['mn']=mon['m'].apply(lambda x: calendar.month_abbr[x])
            st.plotly_chart(px.bar(mon, x='mn', y='pct', title="月效应", color='pct', color_continuous_scale='RdBu_r'), use_container_width=True)

    # === Tab 2: 策略回测 (新增表格) ===
    with tab2:
        with st.expander(f"📖 策略解读：{strategy_mode}", expanded=True):
            st.markdown(STRATEGY_INFO.get(strategy_mode, ""))
            if params['use_trailing']: st.info(f"🛡 **风控设定**：移动止损 (最大回撤 > {params['sl_pct']*100}%) + 止盈 {params['tp_pct']*100}%")
            else: st.warning(f"🛡 **风控设定**：固定止损 (亏损 > {params['sl_pct']*100}%) + 止盈 {params['tp_pct']*100}%")
        
        final_asset, final_bench, ret, bench_ret, trades, df_res = run_strategy_engine(raw_df, strategy_mode, params, initial_cash)
        
        # 指标 (含年化)
        strat_max_dd = (df_res['total_asset']/df_res['total_asset'].cummax()-1).min()
        trade_count = len([t for t in trades if t['type'] == 'Sell'])
        
        win_rate = 0; wl_ratio = 0
        if trade_count > 0:
            wins = [t['pnl'] for t in trades if t['type']=='Sell' and t['pnl']>0]
            losses = [t['pnl'] for t in trades if t['type']=='Sell' and t['pnl']<=0]
            win_rate = len(wins)/trade_count*100
            wl_ratio = np.mean(wins)/abs(np.mean(losses)) if wins and losses else 0
        days = (end_d - start_d).days; cagr = (final_asset/initial_cash)**(365/days)-1 if days>0 else 0
        
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("策略收益", f"{ret*100:.2f}%", f"超额: {(ret-bench_ret)*100:.2f}%")
        c2.metric("年化收益", f"{cagr*100:.2f}%"); c3.metric("最大回撤", f"{strat_max_dd*100:.2f}%")
        c4.metric("胜率", f"{win_rate:.1f}%"); c5.metric("盈亏比", f"{wl_ratio:.2f}"); c6.metric("交易次数", trade_count)
        
        st.subheader("📈 资金曲线")
        st.line_chart(df_res[['total_asset', 'benchmark_asset']].set_axis(['策略', '基准'], axis=1))
        
        st.subheader("🔍 交易详情复盘")
        plot_df = df_res.tail(250)
        
        # 视觉升级版K线图
        fig = go.Figure(data=[go.Candlestick(
            x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'],
            name='K线',
            increasing_line_color='#002FA7', decreasing_line_color='#A9A9A9' # 你的理性奢华配色
        )])
        
        # 画买卖点
        buy_x = [t['date'] for t in trades if t['type']=='Buy' and t['date'] in plot_df.index]
        buy_y = [t['price'] for t in trades if t['type']=='Buy' and t['date'] in plot_df.index]
        sell_x = [t['date'] for t in trades if t['type']=='Sell' and t['date'] in plot_df.index]
        sell_y = [t['price'] for t in trades if t['type']=='Sell' and t['date'] in plot_df.index]
        sell_t = [t['reason'] for t in trades if t['type']=='Sell' and t['date'] in plot_df.index]
        
        fig.add_trace(go.Scatter(mode='markers', x=buy_x, y=buy_y, marker=dict(color='#002FA7', symbol='triangle-up', size=12), name='买入'))
        fig.add_trace(go.Scatter(mode='markers', x=sell_x, y=sell_y, marker=dict(color='#D64045', symbol='triangle-down', size=12), text=sell_t, name='卖出'))
        
        fig.update_layout(height=500, margin=dict(t=30,b=0,l=0,r=0), plot_bgcolor='white', paper_bgcolor='white', xaxis=dict(showgrid=False, linecolor='#e0e0e0'), yaxis=dict(showgrid=True, gridcolor='#f0f0f0'))
        st.plotly_chart(fig, use_container_width=True)

        # === 新增：详细交易表格 ===
        st.subheader("📝 历史交易明细表 (Transaction Log)")
        if len(trades) > 0:
            trade_df = pd.DataFrame(trades)
            trade_df['date'] = trade_df['date'].dt.strftime('%Y-%m-%d')
            trade_df['pnl'] = trade_df['pnl'].apply(lambda x: f"{x*100:.2f}%" if x!=0 else "-")
            trade_df['price'] = trade_df['price'].apply(lambda x: f"{x:.2f}")
            trade_df['total_asset'] = trade_df['total_asset'].apply(lambda x: f"{x:,.0f}")
            
            trade_df.columns = ['日期', '方向', '成交价', '依据', '单笔盈亏', '当前资产']
            st.dataframe(trade_df, use_container_width=True, height=400)
        else:
            st.info("当前回测周期内无交易产生。")

    # === Tab 3: 参数优化 ===
    with tab3:
        st.header("🤖 智能优化"); st.caption(f"区间: {start_d} ~ {end_d}")
        c1, c2 = st.columns(2)
        sl_rg = [i/100 for i in range(int(c1.number_input("止损始",2.0,value=3.0)), int(c1.number_input("止损终",10.0,value=10.0))+1, 1)]
        tp_rg = [i/100 for i in range(int(c2.number_input("止盈始",10.0,value=10.0)), int(c2.number_input("止盈终",50.0,value=50.0))+1, 10)]
        
        if st.button("🚀 优化"):
            bar = st.progress(0); res = []; tp_params = params.copy()
            total_loops = len(sl_rg) * len(tp_rg); counter = 0
            for i, sl in enumerate(sl_rg):
                for tp in tp_rg:
                    tp_params['sl_pct']=sl; tp_params['tp_pct']=tp
                    _,_,r,_,_,_ = run_strategy_engine(raw_df, strategy_mode, tp_params, initial_cash)
                    res.append({'SL':int(sl*100), 'TP':int(tp*100), 'Ret':r})
                    counter += 1; bar.progress(counter / total_loops)
            
            best = pd.DataFrame(res).sort_values('Ret', ascending=False).iloc[0]
            st.success(f"最佳: 止损 {best['SL']}%, 止盈 {best['TP']}%, 收益 {best['Ret']*100:.2f}%")
            
            fig_map = go.Figure(data=go.Heatmap(z=pd.DataFrame(res).pivot(index="SL", columns="TP", values="Ret").values, x=tp_rg, y=sl_rg, colorscale='RdBu', reversescale=True))
            fig_map.update_layout(title="收益率热力图", plot_bgcolor='white')
            st.plotly_chart(fig_map, use_container_width=True)

# --- Footer ---
st.divider()
st.markdown("<div style='text-align: center; color: #888; font-size: 12px;'>Copyright © 2025 <b>Yahui Zhang</b>. All Rights Reserved.</div>", unsafe_allow_html=True)
