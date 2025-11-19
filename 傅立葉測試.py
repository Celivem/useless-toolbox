import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import special, signal  # 引入信號處理庫
import sys
import csv

# 設定 Matplotlib 字型 (避免負號顯示問題)
plt.rcParams['axes.unicode_minus'] = False

def solve_and_plot():
    root = tk.Tk()
    root.withdraw()

    try:
        # ==========================================
        # 1. GUI 輸入階段
        # ==========================================
        print("--- 系統就緒 ---")
        
        prompt_msg = (
            "請輸入函數 f(x) (Python 語法):\n\n"
            "常用範例 (假設區間 -3.14 到 3.14):\n"
            "1. 方波: square(x)\n"
            "2. 多週期方波: square(3 * x)\n"
            "3. 鋸齒波: sawtooth(x)\n"
            "4. 三角波: sawtooth(x, 0.5)\n"
            "5. 傳統正弦: sin(x)"
        )

        func_str = simpledialog.askstring("Input", prompt_msg, initialvalue="square(x)")
        if func_str is None: return

        a_str = simpledialog.askstring("Input", "請輸入區間起始 a:", initialvalue="-3.14159")
        if a_str is None: return
        a = float(a_str)

        b_str = simpledialog.askstring("Input", "請輸入區間結束 b:", initialvalue="3.14159")
        if b_str is None: return
        b = float(b_str)

        n_str = simpledialog.askstring("Input", "請輸入展開項數 N (建議 >= 20):", initialvalue="30")
        if n_str is None: return
        N = int(n_str)

        if N < 1:
            messagebox.showerror("錯誤", "項數 N 必須大於 0")
            return

        # ==========================================
        # 2. 定義數學環境
        # ==========================================
        def f(x_val):
            allowed_locals = {
                "x": x_val,
                "np": np,
                # 基礎數學
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "exp": np.exp, "pi": np.pi, "abs": np.abs, 
                "sqrt": np.sqrt, "log": np.log, "sign": np.sign,
                
                # 波形函數
                "square": signal.square,   
                "sawtooth": signal.sawtooth, 

                # 特殊函數
                "gamma": special.gamma,
                "sinh": np.sinh, "cosh": np.cosh,
            }
            return eval(func_str, {"__builtins__": None}, allowed_locals)

        try:
            f(0.5 * (a + b))
        except Exception as e:
            messagebox.showerror("語法錯誤", f"函數解析失敗: {e}")
            return

        # ==========================================
        # 3. 計算傅立葉係數
        # ==========================================
        L = b - a
        omega = 2 * np.pi / L
        
        print(f"\n正在計算傅立葉係數 (N={N})...")

        excel_data = []
        A_coeffs = []
        B_coeffs = []

        # 計算 A0
        val_a0, _ = quad(lambda x: f(x), a, b, limit=200)
        A0 = (2.0 / L) * val_a0
        A_coeffs.append(A0)
        B_coeffs.append(0.0)
        excel_data.append([0, A0, 0])

        # 計算 An, Bn
        for n in range(1, N + 1):
            val_an, _ = quad(lambda x: f(x) * np.cos(n * omega * x), a, b, limit=100)
            an = (2.0 / L) * val_an
            
            val_bn, _ = quad(lambda x: f(x) * np.sin(n * omega * x), a, b, limit=100)
            bn = (2.0 / L) * val_bn

            A_coeffs.append(an)
            B_coeffs.append(bn)
            excel_data.append([n, an, bn])
            
            if n % 10 == 0: print(f"進度: {n}/{N}")

        # ==========================================
        # 4. 匯出 Excel
        # ==========================================
        save_path = filedialog.asksaveasfilename(
            title="匯出係數表",
            defaultextension=".csv",
            filetypes=[("Excel CSV", "*.csv")],
            initialfile="fourier_data.csv"
        )

        if save_path:
            try:
                with open(save_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t')
                    writer.writerow(['n', 'An', 'Bn'])
                    writer.writerows(excel_data)
                messagebox.showinfo("完成", f"檔案已儲存：{save_path}")
            except Exception as e:
                messagebox.showerror("存檔錯誤", str(e))

        # ==========================================
        # 5. 繪圖邏輯 (依需求修改)
        # ==========================================
        # 增加取樣點讓波形更圓滑
        x_vals = np.linspace(a, b, 2000)
        y_original = np.array([f(val) for val in x_vals])

        def fourier_sum(x_input, k_terms):
            result = A_coeffs[0] / 2.0
            for n in range(1, k_terms + 1):
                result += A_coeffs[n] * np.cos(n * omega * x_input) + \
                          B_coeffs[n] * np.sin(n * omega * x_input)
            return result

        # 計算 N=1, N=3, 以及最終 N
        y_n1 = fourier_sum(x_vals, 1)
        y_n3 = fourier_sum(x_vals, 3)
        y_final = fourier_sum(x_vals, N)

        plt.figure(figsize=(12, 7))
        
        # 1. 原函數 (黑色實線, 粗度 2)
        plt.plot(x_vals, y_original, color='black', linestyle='-', linewidth=2, alpha=0.5, label='Original f(x)')
        
        # 2. N=1 (綠色點線)
        plt.plot(x_vals, y_n1, color='green', linestyle=':', linewidth=1.5, alpha=0.8, label='N=1 (Base)')
        
        # 3. N=3 (橘色點劃線) - 只有當使用者輸入的 N >= 3 時才畫，避免報錯或重疊
        if N >= 3:
            plt.plot(x_vals, y_n3, color='orange', linestyle='-.', linewidth=1.5, alpha=0.8, label='N=3')

        # 4. 最終近似 N (藍色虛線, 粗度 2.5) -> 這是您要求的「改成虛線」
        plt.plot(x_vals, y_final, color='blue', linestyle='--', linewidth=2.5, alpha=0.9, label=f'N={N} Approximation')

        plt.title(f"Fourier Series Analysis: {func_str}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend(loc='upper right', shadow=True) # 圖例加上陰影比較好看
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 標示 x 軸 0 線
        plt.axhline(0, color='black', linewidth=0.8)
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("執行錯誤", str(e))

if __name__ == "__main__":
    solve_and_plot()