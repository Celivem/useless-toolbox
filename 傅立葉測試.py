import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import special, signal
import pandas as pd
import io

# 設定頁面標題與寬度
st.set_page_config(page_title="傅立葉級數視覺化", layout="wide")

# --- 標題與說明 ---
st.title("📈 傅立葉級數線上視覺化 (Fourier Series Viz)")
st.markdown("""
輸入數學函數 $f(x)$，此工具將計算其傅立葉級數近似，並提供 **圖片** 與 **係數表** 下載。
支援語法：`square(x)`, `sawtooth(x)`, `sin(x)`, `abs(x)` 等。
""")

# --- 側邊欄：快速範例選擇 ---
st.sidebar.header("⚡ 快速範例")

# 修正點 1: 這裡移除 'signal.' 前綴，直接呼叫函數名，避免解析錯誤
example_options = {
    "自訂輸入": "",
    "方波 (Square Wave)": "square(x)",
    "多週期方波": "square(3 * x)",
    "鋸齒波 (Sawtooth)": "sawtooth(x)",
    "三角波 (Triangle)": "sawtooth(x, 0.5)",
    "全波整流": "abs(sin(x))",
    "半波整流": "maximum(sin(x), 0)",
    "脈衝波 (Duty Cycle)": "square(x, duty=0.2)"
}

selected_example = st.sidebar.radio("選擇預設波形：", list(example_options.keys()))

# 根據選擇更新預設值
default_func = "x"
if selected_example != "自訂輸入":
    default_func = example_options[selected_example]

# --- 主介面：輸入參數 ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    # 這裡加上 key 以便重置
    func_str = st.text_input("函數 f(x)", value=default_func, help="使用 Python 語法，如 x**2, sin(x)")
with col2:
    a = st.number_input("區間起點 a", value=-3.14159, step=1.0, format="%.4f")
with col3:
    b = st.number_input("區間終點 b", value=3.14159, step=1.0, format="%.4f")
with col4:
    N = st.number_input("展開項數 N", value=30, min_value=1, step=1)

# --- 核心邏輯函數 ---
def get_fourier_data(func_str, a, b, N):
    # 1. 解析函數
    def f(x_val):
        # 修正點 2: 擴充 allowed_locals，確保兼容性
        allowed_locals = {
            "x": x_val, "np": np, "signal": signal,
            # 基礎數學
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "pi": np.pi, "abs": np.abs, 
            "sqrt": np.sqrt, "log": np.log, "sign": np.sign,
            "maximum": np.maximum, "minimum": np.minimum,
            # 信號函數 (直接使用)
            "square": signal.square, "sawtooth": signal.sawtooth,
            # 特殊函數
            "gamma": special.gamma, "sinh": np.sinh, "cosh": np.cosh,
        }
        return eval(func_str, {"__builtins__": None}, allowed_locals)

    # 2. 計算係數
    L = b - a
    omega = 2 * np.pi / L
    
    data = [] 
    A_coeffs = []
    B_coeffs = []

    # A0
    try:
        val_a0, _ = quad(lambda x: f(x), a, b, limit=200)
        A0 = (2.0 / L) * val_a0
    except Exception as e:
        # 捕捉常見錯誤並轉為易讀文字
        return None, None, None, f"解析或積分錯誤: {str(e)}\n請檢查語法 (例如乘號 * 是否遺漏)"

    A_coeffs.append(A0)
    B_coeffs.append(0.0)
    data.append({"n": 0, "An": A0, "Bn": 0.0})

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for n in range(1, N + 1):
        val_an, _ = quad(lambda x: f(x) * np.cos(n * omega * x), a, b, limit=100)
        an = (2.0 / L) * val_an
        
        val_bn, _ = quad(lambda x: f(x) * np.sin(n * omega * x), a, b, limit=100)
        bn = (2.0 / L) * val_bn

        A_coeffs.append(an)
        B_coeffs.append(bn)
        data.append({"n": n, "An": an, "Bn": bn})
        
        if n % 5 == 0:
            progress_bar.progress(n / N)
            status_text.text(f"正在計算第 {n}/{N} 項...")
            
    progress_bar.empty()
    status_text.empty()

    # 3. 準備繪圖函數
    def fourier_sum(x_input, k_terms):
        result = A_coeffs[0] / 2.0
        for k in range(1, k_terms + 1):
            result += A_coeffs[k] * np.cos(k * omega * x_input) + \
                      B_coeffs[k] * np.sin(k * omega * x_input)
        return result

    return data, f, fourier_sum, None

# --- 執行按鈕 ---
if st.button("🚀 開始計算與繪圖", type="primary"):
    with st.spinner("正在進行數學運算..."):
        data_list, f_func, f_sum_func, error_msg = get_fourier_data(func_str, a, b, N)

    if error_msg:
        st.error(error_msg)
    else:
        # 建立 DataFrame
        df = pd.DataFrame(data_list)

        # --- 繪圖區塊 ---
        st.subheader("📊 視覺化結果")
        
        # 設定 Matplotlib
        plt.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_vals = np.linspace(a, b, 1000)
        
        # 繪製原函數
        try:
            y_original = [f_func(val) for val in x_vals]
            ax.plot(x_vals, y_original, 'k-', linewidth=2, alpha=0.5, label='Original f(x)')
        except Exception as e:
            st.warning(f"無法完整繪製原函數: {e}")

        # 繪製近似線
        y_n1 = f_sum_func(x_vals, 1)
        ax.plot(x_vals, y_n1, 'g:', linewidth=1.5, alpha=0.8, label='N=1')

        if N >= 3:
            y_n3 = f_sum_func(x_vals, 3)
            ax.plot(x_vals, y_n3, 'orange', linestyle='-.', linewidth=1.5, alpha=0.8, label='N=3')

        y_final = f_sum_func(x_vals, N)
        ax.plot(x_vals, y_final, 'b--', linewidth=2.5, alpha=0.9, label=f'N={N} Approximation')

        ax.set_title(f"Fourier Series: {func_str}")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(0, color='black', linewidth=0.8)

        st.pyplot(fig)

        # --- 下載區塊 ---
        col_d1, col_d2 = st.columns(2)

        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=300)
        img_buffer.seek(0)
        
        with col_d1:
            st.download_button(
                label="📥 下載圖表 (PNG)",
                data=img_buffer,
                file_name="fourier_plot.png",
                mime="image/png"
            )

        csv_data = df.to_csv(index=False, sep='\t', encoding='utf-8-sig')
        
        with col_d2:
            st.download_button(
                label="📥 下載係數表 (Excel/CSV)",
                data=csv_data,
                file_name="fourier_coefficients.csv",
                mime="text/csv"
            )

        with st.expander("點擊查看詳細係數表"):
            st.dataframe(df)
