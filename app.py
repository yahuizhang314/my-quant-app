import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

# --- 1. 基础配置 ---
st.set_page_config(
    page_title="MyQuant Pro - 由 Yahui Zhang 开发",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 核心计算函数库 ---

@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig', thousands=',')
        df.columns = df.columns.str.strip().str.lower()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        for col in ['close', 'open', 'high', 'low']:
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

# --- 策略引擎 ---
def run_strategy_engine(df_origin, strategy_type, params, initial_cash):
    df = df_origin.copy()
    
    # 预计算指标
    df['MA_S'] = df['close'].rolling(window=params.get('ma_short', 5)).mean()
    df['MA_L'] = df['close'].rolling(window=params.get('ma_long', 20)).mean()
    df['RSI'] = calculate_rsi(df['close'], params.get('rsi_window', 14))
    df['DIF'], df['DEA'], df['MACD_Hist'] = calculate_macd(df, params.get('macd_fast', 12), params.get('macd_slow', 26), params.get('macd_sig', 9))
    df['BB_Up'], df['BB_Mid'], df['BB_Low'] = calculate_bollinger(df, params.get('bb_window', 20), params.get('bb_std', 2))
    
    mom_days = params.get('mom_days', 10)
    df['Momentum'] = df['close'].pct_change(periods=mom_days)

    cash = initial_cash
    shares = 0
    total_assets = []
    trades = [] 
    entry_price = 0
    
    closes = df['close'].values
    dates = df.index
    n = len(df)
    
    ind_ma_s = df['MA_S'].values
    ind_ma_l = df['MA_L'].values
    ind_rsi = df['RSI'].values
    ind_dif = df['DIF'].values
    ind_dea = df['DEA'].values
    ind_bb_low = df['BB_Low'].values
    ind_bb_up = df['BB_Up'].values
    ind_mom = df['Momentum'].values
    
    sl_pct = params.get('sl_pct', 0.05)
    tp_pct = params.get('tp_pct', 0.20)
    
    for i in range(n):
        price = closes[i]
        current_date = dates[i]
        
        if shares == 0:
            buy_signal = False
            if strategy_type == "双均线 (Dual MA)":
                if not np.isnan(ind_ma_s[i]) and not np.isnan(ind_ma_l[i]):
                    if ind_ma_s[i] > ind_ma_l[i]: buy_signal = True
            elif strategy_type == "RSI 反转 (RSI Reversion)":
                if not np.isnan(ind_rsi[i]) and ind_rsi[i] < params['rsi_buy']: buy_signal = True
            elif strategy_type == "MACD 趋势 (MACD Trend)":
                if i > 0 and not np.isnan(ind_dif[i]) and ind_dif[i] > ind_dea[i] and ind_dif[i-1] <= ind_dea[i-1]: buy_signal = True
            elif strategy_type == "布林带回归 (Bollinger)":
                if not np.isnan(ind_bb_low[i]) and price < ind_bb_low[i]: buy_signal = True
            elif strategy_type == "动量突破 (Momentum)":
                if not np.isnan(ind_mom[i]) and ind_mom[i] > params['mom_thres']: buy_signal = True

            if buy_signal:
                shares = int(cash / price)
                cash -= shares * price
                entry_price = price
                trades.append([current_date, "Buy", price, 0])

        else:
            sell_type = None
            pct_change = (price - entry_price) / entry_price
            
            if pct_change <= -sl_pct: sell_type = "止损 🛑"
            elif pct_change >= tp_pct: sell_type = "止盈 💰"
            
            elif strategy_type == "双均线 (Dual MA)" and ind_ma_s[i] < ind_ma_l[i]: sell_type = "死叉离场"
            elif strategy_type == "RSI 反转 (RSI Reversion)" and ind_rsi[i] > params['rsi_sell']: sell_type = "RSI高位离场"
            elif strategy_type == "MACD 趋势 (MACD Trend)" and ind_dif[i] < ind_dea[i]: sell_type = "MACD死叉"
            elif strategy_type == "布林带回归 (Bollinger)" and price > ind_bb_up[i]: sell_type = "触顶回调"
            elif strategy_type == "动量突破 (Momentum)" and ind_mom[i] < 0: sell_type = "动量消失"
            
            if sell_type:
                cash += shares * price
                shares = 0
                trades.append([current_date, sell_type, price, pct_change])
        
        total_assets.append(cash + shares * price)
    
    # === 修复点：确保变量名一致 ===
    df['total_asset'] = total_assets
    first_price = df['close'].iloc[0]
    df['benchmark_asset'] = initial_cash * (df['close'] / first_price)
    
    final_asset = total_assets[-1]
    final_benchmark = df['benchmark_asset'].iloc[-1] # 修复：统一变量名为 final_benchmark
    
    ret = (final_asset - initial_cash) / initial_cash
    bench_ret = (final_benchmark - initial_cash) / initial_cash
    
    return final_asset, final_benchmark, ret, bench_ret, trades, df

# --- 3. 侧边栏 ---
with st.sidebar:
    st.header("🎛 策略控制台")
    current_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not current_files:
        st.error("请放入 CSV 文件")
        st.stop()
    selected_file = st.selectbox("1. 选择标的", current_files)
    raw_df = load_data(selected_file)
    st.divider()
    st.subheader("2. 策略模型")
    strategies = ["双均线 (Dual MA)", "MACD 趋势 (MACD Trend)", "动量突破 (Momentum)", "RSI 反转 (RSI Reversion)", "布林带回归 (Bollinger)"]
    strategy_mode = st.selectbox("选择基准策略", strategies)
    params = {}
    initial_cash = st.number_input("初始资金", value=1000000, step=10000)
    
    st.caption("--- 策略参数 ---")
    if strategy_mode == "双均线 (Dual MA)":
        params['ma_short'] = st.slider("短期均线", 2, 60, 5)
        params['ma_long'] = st.slider("长期均线", 10, 120, 20)
    elif strategy_mode == "MACD 趋势 (MACD Trend)":
        params['macd_fast'] = st.number_input("快线", value=12)
        params['macd_slow'] = st.number_input("慢线", value=26)
        params['macd_sig'] = st.number_input("信号", value=9)
    elif strategy_mode == "动量突破 (Momentum)":
        params['mom_days'] = st.slider("动量周期", 5, 60, 20)
        params['mom_thres'] = st.slider("追涨阈值 (%)", 0.0, 10.0, 2.0) / 100
    elif strategy_mode == "RSI 反转 (RSI Reversion)":
        params['rsi_window'] = st.slider("RSI周期", 6, 24, 14)
        params['rsi_buy'] = st.slider("买入阈值", 10, 40, 30)
        params['rsi_sell'] = st.slider("卖出阈值", 60, 90, 70)
    elif strategy_mode == "布林带回归 (Bollinger)":
        params['bb_window'] = st.slider("周期", 10, 60, 20)
        params['bb_std'] = st.slider("标准差", 1.0, 3.0, 2.0)
        
    st.caption("--- 风控参数 ---")
    c_sl, c_tp = st.columns(2)
    params['sl_pct'] = c_sl.slider("止损(%)", 1.0, 20.0, 5.0) / 100
    params['tp_pct'] = c_tp.slider("止盈(%)", 5.0, 100.0, 20.0) / 100

# --- 4. 主界面 ---
if raw_df is not None:
    st.title(f"📊 量化工作台: {selected_file}")
    
    # 顶部添加作者信息
    st.caption(f"Developed by **Yahui Zhang** | MyQuant Pro v10.1")

    tab1, tab2, tab3 = st.tabs(["🔍 市场体检", "⚔️ 策略 vs 基准", "🤖 参数优化"])
    
    # === Tab 1 ===
    with tab1:
        st.subheader("1. 市场性格分析")
        roll_max = raw_df['close'].cummax()
        daily_dd = raw_df['close'] / roll_max - 1.0
        max_dd = daily_dd.min()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("总交易天数", len(raw_df))
        k2.metric("当前价格", raw_df['close'].iloc[-1])
        k3.metric("区间总涨幅", f"{((raw_df['close'].iloc[-1]/raw_df['close'].iloc[0])-1)*100:.2f}%")
        k4.metric("历史最大回撤", f"{max_dd*100:.2f}%")
        st.divider()
        st.subheader("📅 日历效应")
        df_cal = raw_df.copy()
        df_cal['pct'] = df_cal['close'].pct_change() * 100
        df_cal['weekday_name'] = df_cal.index.day_name()
        df_cal['weekday_idx'] = df_cal.index.dayofweek
        cal_stats = df_cal.groupby(['weekday_idx', 'weekday_name'])['pct'].mean().reset_index()
        cal_stats.sort_values('weekday_idx', inplace=True)
        fig_cal = px.bar(cal_stats, x='weekday_name', y='pct', title="周一至周五平均涨跌幅 (%)", color='pct', color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_cal, use_container_width=True)

    # === Tab 2 ===
    with tab2:
        st.subheader(f"🚀 策略表现 vs 买入持有 ({strategy_mode})")
        final_asset, final_bench, ret, bench_ret, trades, df_res = run_strategy_engine(raw_df, strategy_mode, params, initial_cash)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("策略收益率", f"{ret*100:.2f}%", f"¥{final_asset - initial_cash:,.0f}")
        m2.metric("基准(买入持有)", f"{bench_ret*100:.2f}%", f"¥{final_bench - initial_cash:,.0f}")
        alpha = ret - bench_ret
        m3.metric("超额收益 (Alpha)", f"{alpha*100:.2f}%")
        trade_count = len([t for t in trades if t[1]!='Buy'])
        m4.metric("交易次数", trade_count)
        
        st.subheader("📈 资金曲线对比")
        chart_data = df_res[['total_asset', 'benchmark_asset']].copy()
        chart_data.columns = ['我的策略 (Strategy)', '基准指数 (Benchmark)']
        st.line_chart(chart_data)
        
        st.subheader("🔍 买卖点详情")
        plot_df = df_res.tail(250)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], name='K线'))
        buy_x = [t[0] for t in trades if t[1]=='Buy' and t[0] in plot_df.index]
        buy_y = [t[2] for t in trades if t[1]=='Buy' and t[0] in plot_df.index]
        sell_x = [t[0] for t in trades if t[1]!='Buy' and t[0] in plot_df.index]
        sell_y = [t[2] for t in trades if t[1]!='Buy' and t[0] in plot_df.index]
        sell_t = [t[1] for t in trades if t[1]!='Buy' and t[0] in plot_df.index]
        fig.add_trace(go.Scatter(mode='markers', x=buy_x, y=buy_y, marker=dict(color='green', symbol='triangle-up', size=12), name='买入'))
        fig.add_trace(go.Scatter(mode='markers', x=sell_x, y=sell_y, marker=dict(color='red', symbol='triangle-down', size=12), text=sell_t, name='卖出'))
        fig.update_layout(height=500, xaxis_title="日期", yaxis_title="价格")
        st.plotly_chart(fig, use_container_width=True)

    # === Tab 3 ===
    with tab3:
        st.header("🤖 网格搜索 (Grid Search)")
        st.info("寻找收益率最高的止损/止盈组合")
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            sl_start = st.number_input("止损起始 (%)", 2.0, value=3.0)
            sl_end = st.number_input("止损结束 (%)", 5.0, value=10.0)
            sl_step = st.number_input("止损步长", 1.0, value=1.0)
        with col_opt2:
            tp_start = st.number_input("止盈起始 (%)", 10.0, value=10.0)
            tp_end = st.number_input("止盈结束 (%)", 50.0, value=50.0)
            tp_step = st.number_input("止盈步长", 5.0, value=10.0)

        if st.button("🚀 开始优化"):
            sl_range = [i/100 for i in range(int(sl_start), int(sl_end)+1, int(sl_step))]
            tp_range = [i/100 for i in range(int(tp_start), int(tp_end)+1, int(tp_step))]
            total_steps = len(sl_range) * len(tp_range)
            progress_bar = st.progress(0)
            results = []
            test_params = params.copy()
            counter = 0
            for sl in sl_range:
                for tp in tp_range:
                    test_params['sl_pct'] = sl
                    test_params['tp_pct'] = tp
                    _, _, ret, _, _, _ = run_strategy_engine(raw_df, strategy_mode, test_params, initial_cash)
                    results.append({"止损(%)": int(sl*100), "止盈(%)": int(tp*100), "收益率": ret})
                    counter += 1
                    progress_bar.progress(counter / total_steps)
            
            res_df = pd.DataFrame(results)
            best_res = res_df.loc[res_df['收益率'].idxmax()]
            st.success(f"最优组合：止损 {best_res['止损(%)']}%，止盈 {best_res['止盈(%)']}%，收益率 {best_res['收益率']*100:.2f}%")
            heatmap_data = res_df.pivot(index="止损(%)", columns="止盈(%)", values="收益率")
            fig_map = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index, colorscale='RdBu', reversescale=True))
            fig_map.update_layout(title="收益率热力图", xaxis_title="止盈 (%)", yaxis_title="止损 (%)")
            st.plotly_chart(fig_map, use_container_width=True)

# --- 5. 页脚署名 (Footer) ---
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #888888; padding: 20px;'>
        <p>Copyright © 2025 <b>Yahui Zhang</b>. All Rights Reserved.</p>
        <p style='font-size: 12px;'>Powered by Python & Streamlit </p>
    </div>
    """,
    unsafe_allow_html=True
)
