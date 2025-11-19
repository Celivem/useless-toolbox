import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import special, signal
import pandas as pd
import io

# 設定頁面
st.set_page_config(page_title="傅立葉級數視覺化 (互動版)", layout="wide")

# --- 初始化 Session State (用來暫存計算結果) ---
if 'fourier_result' not in st.session_state:
    st.session_state['fourier_result'] = None

# --- 標題 ---
st.title("📈 傅立葉級數互動實驗室")
st.markdown("""
1. 設定 **最大項數 (Max N)** 並按下計算。
2. 計算完成後，使用下方的 **拉桿** 即時調整 N 值，觀察波形如何逼近。
""")

# --- 側邊欄：快速範例 ---
st.sidebar.header("⚡ 快速範例")
example_options = {
    "自訂輸入": "",
    "方波 (Square)": "square(x)",
    "多週期方波": "square(3 * x)",
    "鋸齒波 (Sawtooth)": "sawtooth(x)",
    "三角波": "sawtooth(x, 0.5)",
    "全波整流": "abs(sin(x))",
    "半波整流": "maximum(sin(x), 0)",
    "脈衝波": "square(x, duty=0.2)"
}
selected_example = st.sidebar.radio("選擇預設波形：", list(example_options.keys()))
default_func = "square(x)" if selected_example != "自訂輸入" else "x"
if selected_example != "自訂輸入":
    default_func = example_options[selected_example]

# --- 參數設定區 ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    func_str = st.text_input("函數 f(x)", value=default_func)
with col2:
    a = st.number_input("區間起點 a", value=-3.1415, step=1.0, format="%.4f")
with col3:
    b = st.number_input("區間終點 b", value=3.1415, step=1.0, format="%.4f")
with col4:
    # 這裡改名為 Max N，代表計算的上限
    max_n = st.number_input("最大項數 (計算上限)", value=50, min_value=1, step=10)

# --- 核心運算函數 (一次算完所有係數) ---
def calculate_coefficients(func_str, a, b, max_n):
    # 1. 解析函數
    def f(x_val):
        allowed_locals = {
            "x": x_val, "np": np, "signal": signal,
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "pi": np.pi, "abs": np.abs, 
            "sqrt": np.sqrt, "log": np.log, "sign": np.sign,
            "maximum": np.maximum, "minimum": np.minimum,
            "square": signal.square, "sawtooth": signal.sawtooth,
            "gamma": special.gamma, "sinh": np.sinh, "cosh": np.cosh,
        }
        return eval(func_str, {"__builtins__": None}, allowed_locals)

    L = b - a
    omega = 2 * np.pi / L
    
    A_coeffs = []
    B_coeffs = []
    
    # 進度條
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 計算 A0
    try:
        val_a0, _ = quad(lambda x: f(x), a, b, limit=200)
        A0 = (2.0 / L) * val_a0
    except Exception as e:
        return None, f"積分錯誤: {str(e)}"

    A_coeffs.append(A0)
    B_coeffs.append(0.0)

    # 計算 An, Bn (直到 Max N)
    for n in range(1, max_n + 1):
        val_an, _ = quad(lambda x: f(x) * np.cos(n * omega * x), a, b, limit=100)
        an = (2.0 / L) * val_an
        
        val_bn, _ = quad(lambda x: f(x) * np.sin(n * omega * x), a, b, limit=100)
        bn = (2.0 / L) * val_bn

        A_coeffs.append(an)
        B_coeffs.append(bn)

        if n % 5 == 0:
            progress_bar.progress(n / max_n)
            status_text.text(f"正在計算係數: {n}/{max_n}")

    progress_bar.empty()
    status_text.empty()

    # 為了加速繪圖，我們先算出原函數的 y 值存起來
    x_vals = np.linspace(a, b, 1000)
    try:
        y_original = [f(val) for val in x_vals]
    except:
        y_original = None

    # 將結果打包回傳
    return {
        "A": A_coeffs,
        "B": B_coeffs,
        "omega": omega,
        "x_vals": x_vals,
        "y_original": y_original,
        "func_str": func_str,
        "L": L,
        "range": (a, b)
    }, None

# --- 按鈕區 ---
if st.button("🚀 開始計算 (建立係數庫)", type="primary"):
    with st.spinner("正在進行積分運算，這可能需要一點時間..."):
        result, error = calculate_coefficients(func_str, a, b, max_n)
        
    if error:
        st.error(error)
    else:
        # 將結果存入 Session State，這樣拉動拉桿時才不會重算
        st.session_state['fourier_result'] = result
        st.rerun() # 重新整理頁面以顯示拉桿

# --- 結果顯示區 (只有當計算過後才會出現) ---
if st.session_state['fourier_result'] is not None:
    res = st.session_state['fourier_result']
    
    st.divider()
    
    # === 互動拉桿區 ===
    # 這裡的拉桿變動時，因為我們用的是 session_state 的數據，所以反應會極快
    current_n = st.slider(
        "🎚️ 調整 N 值 (觀察逼近過程)", 
        min_value=0, 
        max_value=len(res["A"]) - 1, 
        value=min(10, len(res["A"]) - 1)
    )

    # === 快速合成函數 ===
    # 利用 numpy 向量運算，不做積分，速度極快
    def fast_reconstruct(n_terms):
        # S = A0/2
        y_approx = np.full_like(res["x_vals"], res["A"][0] / 2.0)
        # + Sum(An cos + Bn sin)
        for k in range(1, n_terms + 1):
            y_approx += res["A"][k] * np.cos(k * res["omega"] * res["x_vals"]) + \
                        res["B"][k] * np.sin(k * res["omega"] * res["x_vals"])
        return y_approx

    # 計算當前 N 的波形
    y_current = fast_reconstruct(current_n)

    # === 繪圖 ===
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 1. 原函數 (黑線)
    if res["y_original"] is not None:
        ax.plot(res["x_vals"], res["y_original"], 'k-', linewidth=2, alpha=0.4, label='Original')

    # 2. 當前 N 的近似 (藍線)
    ax.plot(res["x_vals"], y_current, 'b-', linewidth=2.5, alpha=0.9, label=f'N={current_n}')
    
    # 3. N=1 基頻 (參考用，綠虛線)
    if current_n > 1:
        y_n1 = fast_reconstruct(1)
        ax.plot(res["x_vals"], y_n1, 'g:', alpha=0.6, linewidth=1, label='N=1')

    ax.set_title(f"Fourier Series Approximation (N={current_n})")
    ax.set_ylim(np.min(y_current)*1.2 - 1, np.max(y_current)*1.2 + 1) # 固定 Y 軸避免跳動
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig)

    # === 下載區 ===
    col_d1, col_d2 = st.columns(2)
    
    # 圖片下載
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300)
    img_buffer.seek(0)
    col_d1.download_button("📥 下載此圖 (PNG)", img_buffer, f"fourier_N{current_n}.png", "image/png")

    # 表格下載 (產生包含所有係數的表)
    df = pd.DataFrame({
        "n": range(len(res["A"])),
        "An": res["A"],
        "Bn": res["B"]
    })
    csv_data = df.to_csv(index=False, sep='\t', encoding='utf-8-sig')
    col_d2.download_button("📥 下載完整係數表 (CSV)", csv_data, "coeffs.csv", "text/csv")

    # 係數預覽
    with st.expander(f"查看前 {current_n} 項係數數值"):
        st.dataframe(df.head(current_n + 1))

    # 重置按鈕
    if st.button("🔄 清除結果 / 重新輸入"):
        st.session_state['fourier_result'] = None
        st.rerun()
