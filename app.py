import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px # 引入更高级的统计绘图库
import os

# --- 1. 基础配置 ---
st.set_page_config(page_title="MyQuant Pro - 全功能量化系统", layout="wide")

# --- 2. 核心逻辑函数库 (不含任何界面展示代码，只做计算) ---

@st.cache_data
def load_data(file_path):
    """读取并清洗数据"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig', thousands=',')
        df.columns = df.columns.str.strip().str.lower()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # 强制转数字
        for col in ['close', 'open', 'high', 'low']:
             df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['close'], inplace=True)
        return df
    except Exception as e:
        return None

def calculate_max_drawdown(df):
    """计算最大回撤"""
    # 历史最高价
    roll_max = df['close'].cummax()
    # 每日回撤幅度
    daily_dd = df['close'] / roll_max - 1.0
    # 最大回撤
    max_dd = daily_dd.min()
    return max_dd, daily_dd

def run_strategy_logic(df_origin, ma_days, sl_pct, tp_pct, initial_cash):
    """
    纯粹的策略逻辑，没有任何绘图代码。
    是为了方便在循环中多次调用，跑得更快。
    """
    df = df_origin.copy()
    
    # 计算均线
    df['MA'] = df['close'].rolling(window=ma_days).mean()
    
    cash = initial_cash
    shares = 0
    total_assets = []
    
    entry_price = 0
    
    # 这种循环在Python里其实效率不高，但为了逻辑清晰（止盈止损需要逐日判断），暂时这样做
    # 进阶做法是用 vectorbt 等库，但这超出了初学者范围
    
    # 预先计算好数据，减少循环内的pandas操作
    closes = df['close'].values
    mas = df['MA'].values
    dates = df.index
    
    n = len(df)
    
    # 记录交易
    trades = [] # 记录每一笔交易 [日期, 类型, 价格, 盈亏比例]
    
    for i in range(n):
        price = closes[i]
        ma_val = mas[i]
        current_date = dates[i]
        
        # 如果均线是NaN（前几天），直接跳过
        if pd.isna(ma_val):
            total_assets.append(cash)
            continue
            
        # --- 策略逻辑 ---
        if shares == 0:
            # 买入信号：收盘价 > 均线
            if price > ma_val:
                shares = int(cash / price)
                cash -= shares * price
                entry_price = price
                trades.append([current_date, "Buy", price, 0])
        else:
            sell_type = None
            pct_change = (price - entry_price) / entry_price
            
            # 止损
            if pct_change <= -sl_pct:
                sell_type = "止损"
            # 止盈
            elif pct_change >= tp_pct:
                sell_type = "止盈"
            # 趋势破位
            elif price < ma_val:
                sell_type = "趋势破位"
            
            if sell_type:
                cash += shares * price
                shares = 0
                trades.append([current_date, sell_type, price, pct_change])
        
        total_assets.append(cash + shares * price)
        
    df['total_asset'] = total_assets
    
    # 计算最终指标
    final_asset = total_assets[-1]
    ret = (final_asset - initial_cash) / initial_cash
    
    return final_asset, ret, trades, df

# --- 3. 界面侧边栏 ---
with st.sidebar:
    st.header("🎛 控制面板")
    
    # 文件选择
    current_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not current_files:
        st.error("没有找到CSV文件")
        selected_file = None
    else:
        selected_file = st.selectbox("选择数据源", current_files)
        
    if selected_file:
        # 读取数据 (只读一次，放入缓存)
        raw_df = load_data(selected_file)
        if raw_df is None:
            st.error("数据读取失败")
            st.stop()
            
    st.divider()
    initial_cash = st.number_input("初始资金", value=1000000, step=10000)


# --- 4. 主界面：三板块架构 ---

if selected_file and raw_df is not None:
    st.title(f"📊 量化分析报告: {selected_file}")
    
    # 创建三个标签页
    tab1, tab2, tab3 = st.tabs(["🔍 数据体检", "📈 策略回测", "🚀 寻找最优参"])

    # ==========================================
    # Tab 1: 数据体检 (Data Discovery)
    # ==========================================
    with tab1:
        st.subheader("1. 基础概况")
        # 计算最大回撤
        max_dd, dd_series = calculate_max_drawdown(raw_df)
        total_ret = (raw_df['close'].iloc[-1] / raw_df['close'].iloc[0]) - 1
        days = len(raw_df)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总交易天数", f"{days} 天")
        c2.metric("区间涨跌幅", f"{total_ret*100:.2f}%")
        c3.metric("历史最大回撤", f"{max_dd*100:.2f}%", help="历史上最惨的一次是从最高点跌了多少")
        c4.metric("最高/最低价", f"{raw_df['high'].max()} / {raw_df['low'].min()}")
        
        st.subheader("2. 价格分布与波动")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("**日涨跌幅分布 (直方图)**")
            # 计算日涨跌幅
            daily_pct = raw_df['close'].pct_change() * 100
            fig_hist = px.histogram(daily_pct, x="close", nbins=50, title="涨跌幅分布 (看是否有肥尾风险)")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_chart2:
            st.markdown("**历史回撤走势 (潜水图)**")
            st.area_chart(dd_series, color="#ff4b4b")
            st.caption("红色区域越深，代表当时被套得越惨")

    # ==========================================
    # Tab 2: 单次策略回测 (Manual Backtest)
    # ==========================================
    with tab2:
        st.markdown("### 手动调整参数查看结果")
        
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            ma_days = st.slider("均线周期", 5, 120, 20, key="t2_ma")
        with col_p2:
            sl_pct = st.slider("止损阈值 (%)", 1.0, 20.0, 5.0, step=0.5, key="t2_sl") / 100
        with col_p3:
            tp_pct = st.slider("止盈阈值 (%)", 5.0, 100.0, 20.0, step=2.0, key="t2_tp") / 100

        # 运行按钮
        if st.button("开始回测", type="primary"):
            final_asset, ret, trades, df_res = run_strategy_logic(raw_df, ma_days, sl_pct, tp_pct, initial_cash)
            
            # 展示结果
            k1, k2, k3 = st.columns(3)
            k1.metric("策略收益率", f"{ret*100:.2f}%", delta=f"{final_asset-initial_cash:,.0f}")
            k2.metric("交易次数", f"{len([t for t in trades if t[1]!='Buy'])} 次")
            k3.metric("最终资产", f"¥{final_asset:,.0f}")
            
            # 画资金曲线
            st.line_chart(df_res['total_asset'])
            
            # 交易明细
            st.write("交易记录摘要：")
            trade_df = pd.DataFrame(trades, columns=["日期", "操作", "价格", "盈亏比例"])
            # 格式化一下显示
            trade_df['盈亏比例'] = trade_df['盈亏比例'].apply(lambda x: f"{x*100:.2f}%" if x!=0 else "-")
            st.dataframe(trade_df, height=300)

    # ==========================================
    # Tab 3: 参数优化 (Optimization / Grid Search)
    # ==========================================
    with tab3:
        st.markdown("### 🤖 寻找最优的止盈止损组合")
        st.info("原理：电脑会自动尝试所有你指定的组合，这可能需要一点时间。")
        
        # 优化参数设置
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            st.markdown("**止损范围设置 (Stop Loss)**")
            sl_start = st.number_input("止损起始 (%)", 2.0, value=3.0)
            sl_end = st.number_input("止损结束 (%)", 5.0, value=10.0)
            sl_step = st.number_input("步长", 1.0, value=1.0)
            
        with col_o2:
            st.markdown("**止盈范围设置 (Take Profit)**")
            tp_start = st.number_input("止盈起始 (%)", 10.0, value=10.0)
            tp_end = st.number_input("止盈结束 (%)", 50.0, value=50.0)
            tp_step = st.number_input("步长", 5.0, value=10.0)
            
        ma_fixed = st.slider("固定均线周期 (作为基准)", 5, 120, 20, key="t3_ma")

        if st.button("🚀 启动网格搜索"):
            # 生成参数列表
            sl_range = [i/100 for i in range(int(sl_start), int(sl_end)+1, int(sl_step))]
            tp_range = [i/100 for i in range(int(tp_start), int(tp_end)+1, int(tp_step))]
            
            # 进度条
            total_steps = len(sl_range) * len(tp_range)
            progress_bar = st.progress(0)
            step_counter = 0
            
            results = [] # 存储结果
            
            # 双重循环 (Grid Search)
            for sl in sl_range:
                for tp in tp_range:
                    # 运行策略
                    _, ret, _, _ = run_strategy_logic(raw_df, ma_fixed, sl, tp, initial_cash)
                    
                    results.append({
                        "止损(%)": int(sl*100),
                        "止盈(%)": int(tp*100),
                        "收益率": ret
                    })
                    
                    # 更新进度条
                    step_counter += 1
                    progress_bar.progress(step_counter / total_steps)
            
            # 结果转为 DataFrame
            res_df = pd.DataFrame(results)
            
            # 找出最优
            best_res = res_df.loc[res_df['收益率'].idxmax()]
            st.success(f"✅ 搜索完成！最优组合：止损 {best_res['止损(%)']}%，止盈 {best_res['止盈(%)']}%，收益率 {best_res['收益率']*100:.2f}%")
            
            # --- 核心可视化：热力图 (Heatmap) ---
            st.subheader("🔥 参数热力图 (颜色越红越赚钱)")
            
            # 使用 Plotly 画热力图
            # Pivot table 把数据变成矩阵格式
            heatmap_data = res_df.pivot(index="止损(%)", columns="止盈(%)", values="收益率")
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='RdBu', # 红蓝配色，红赚钱，蓝亏钱
                reversescale=True, # 让红色代表高数值
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=f"收益率分布图 (固定均线 {ma_fixed}日)",
                xaxis_title="止盈设置 (%)",
                yaxis_title="止损设置 (%)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(res_df.sort_values(by="收益率", ascending=False).head(10))

else:
    st.info("👈 请先在左侧上传或选择数据文件。")