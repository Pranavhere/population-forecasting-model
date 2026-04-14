"""
Atlanta MSA Population Forecast
Model 1: Basic Component Extrapolation

Requires: Final_Data_for_Modeling.xlsx with sheet 'pop_estimate_components'
Outputs:  6 forecast charts (POP, BIRTHS, DEATHS, INT_MIG, DOM_MIG, RESIDUAL)
          component_projection_model DataFrame
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 0. LOAD DATA
# ============================================================
excel_file = '/Users/pranavlakhotia/Downloads/Final_Data_for_Modeling.xlsx'
xls = pd.ExcelFile(excel_file)

pop_estimate_components = pd.read_excel(excel_file, sheet_name='pop_estimate_components')
pop_by_agesex           = pd.read_excel(excel_file, sheet_name='pop_by_agesex')
mortality_rate_2020_census = pd.read_excel(excel_file, sheet_name='mortality_rate_2020_census')

# ============================================================
# 1. PREPARE TIME SERIES
# ============================================================
df = pop_estimate_components.copy()
df = df.sort_values("YEAR").reset_index(drop=True)

# ============================================================
# 2. CONVERT COUNTS TO RATES
#    b_t = B_t/P_t,  d_t = D_t/P_t,  etc.
# ============================================================
df["birth_rate"]   = df["BIRTHS"]           / df["POP_ESTIMATE"]
df["death_rate"]   = df["DEATHS"]           / df["POP_ESTIMATE"]
df["int_mig_rate"] = df["INTERNATIONAL_MIG"] / df["POP_ESTIMATE"]
df["dom_mig_rate"] = df["DOMESTIC_MIG"]     / df["POP_ESTIMATE"]
df["res_rate"]     = df["RESIDUAL"]         / df["POP_ESTIMATE"]

# ============================================================
# 3. ROLLING WINDOW ESTIMATES (10-year window)
#    mu_r = rolling mean,  sigma_r = rolling std
# ============================================================
window = 10
rates  = ["birth_rate", "death_rate", "int_mig_rate", "dom_mig_rate", "res_rate"]

means = {}
stds  = {}

for r in rates:
    means[r] = df[r].rolling(window).mean().iloc[-1]
    stds[r]  = df[r].rolling(window).std().iloc[-1]

print("Rolling-window rate estimates (last 10 years):")
for r in rates:
    print(f"  {r:18s}  mean={means[r]:.6f}  std={stds[r]:.6f}")

# ============================================================
# 4. TOTAL GROWTH RATE AND UNCERTAINTY
#    g = b - d + m_int + m_dom + r
#    sigma_g = sqrt(sigma_b^2 + sigma_d^2 + ...)
# ============================================================
var_total = sum(stds[r]**2 for r in rates)
sigma_g   = np.sqrt(var_total)
g         = (means["birth_rate"]
             - means["death_rate"]
             + means["int_mig_rate"]
             + means["dom_mig_rate"]
             + means["res_rate"])

print(f"\nTotal growth rate g      = {g:.6f}")
print(f"Total uncertainty sigma_g = {sigma_g:.6f}")

# ============================================================
# 5. PROJECTION LOOP  (10 years forward)
#    P_{t+1} = P_t * (1 + g)
#    sigma_t = sigma_g * sqrt(t)
# ============================================================
forecast_years = 10
last_year      = df["YEAR"].iloc[-1]
last_pop       = df["POP_ESTIMATE"].iloc[-1]

results      = []
current_pop  = last_pop

for i in range(1, forecast_years + 1):
    year     = last_year + i
    next_pop = current_pop * (1 + g)

    # Time-dependent uncertainty
    sigma_t = sigma_g * np.sqrt(i)

    # ------ Population confidence intervals ------
    pop_1s_up  = next_pop * (1 + sigma_t)
    pop_1s_low = next_pop * (1 - sigma_t)
    pop_2s_up  = next_pop * (1 + 2 * sigma_t)
    pop_2s_low = next_pop * (1 - 2 * sigma_t)

    # ------ Component point estimates ------
    births   = next_pop * means["birth_rate"]
    deaths   = next_pop * means["death_rate"]
    int_mig  = next_pop * means["int_mig_rate"]
    dom_mig  = next_pop * means["dom_mig_rate"]
    residual = next_pop * means["res_rate"]

    # ------ Component uncertainty ------
    births_std   = births   * stds["birth_rate"]   * np.sqrt(i)
    deaths_std   = deaths   * stds["death_rate"]   * np.sqrt(i)
    int_mig_std  = int_mig  * stds["int_mig_rate"] * np.sqrt(i)
    dom_mig_std  = dom_mig  * stds["dom_mig_rate"] * np.sqrt(i)
    residual_std = residual * stds["res_rate"]      * np.sqrt(i)

    results.append({
        "YEAR":          year,
        "POP_ESTIMATE":  next_pop,
        "POP_1S_UP":     pop_1s_up,
        "POP_1S_LOW":    pop_1s_low,
        "POP_2S_UP":     pop_2s_up,
        "POP_2S_LOW":    pop_2s_low,

        "BIRTHS":        births,
        "BIRTHS_1S_UP":  births + births_std,
        "BIRTHS_1S_LOW": births - births_std,
        "BIRTHS_2S_UP":  births + 2 * births_std,
        "BIRTHS_2S_LOW": births - 2 * births_std,

        "DEATHS":        deaths,
        "DEATHS_1S_UP":  deaths + deaths_std,
        "DEATHS_1S_LOW": deaths - deaths_std,
        "DEATHS_2S_UP":  deaths + 2 * deaths_std,
        "DEATHS_2S_LOW": deaths - 2 * deaths_std,

        "INT_MIG":        int_mig,
        "INT_MIG_1S_UP":  int_mig + int_mig_std,
        "INT_MIG_1S_LOW": int_mig - int_mig_std,
        "INT_MIG_2S_UP":  int_mig + 2 * int_mig_std,
        "INT_MIG_2S_LOW": int_mig - 2 * int_mig_std,

        "DOM_MIG":        dom_mig,
        "DOM_MIG_1S_UP":  dom_mig + dom_mig_std,
        "DOM_MIG_1S_LOW": dom_mig - dom_mig_std,
        "DOM_MIG_2S_UP":  dom_mig + 2 * dom_mig_std,
        "DOM_MIG_2S_LOW": dom_mig - 2 * dom_mig_std,

        "RESIDUAL":    residual,
        "RES_1S_UP":   residual + residual_std,
        "RES_1S_LOW":  residual - residual_std,
        "RES_2S_UP":   residual + 2 * residual_std,
        "RES_2S_LOW":  residual - 2 * residual_std,
    })

    current_pop = next_pop

component_projection_model = pd.DataFrame(results)
print("\nProjection complete. Preview:")
print(component_projection_model[["YEAR","POP_ESTIMATE","POP_1S_LOW","POP_1S_UP"]].to_string(index=False))

# ============================================================
# 6. PLOTS  — one figure per component
# ============================================================
plot_config = {
    "POP_ESTIMATE": ("POP_1S_LOW", "POP_1S_UP", "POP_2S_LOW", "POP_2S_UP"),
    "BIRTHS":       ("BIRTHS_1S_LOW", "BIRTHS_1S_UP", "BIRTHS_2S_LOW", "BIRTHS_2S_UP"),
    "DEATHS":       ("DEATHS_1S_LOW", "DEATHS_1S_UP", "DEATHS_2S_LOW", "DEATHS_2S_UP"),
    "INT_MIG":      ("INT_MIG_1S_LOW", "INT_MIG_1S_UP", "INT_MIG_2S_LOW", "INT_MIG_2S_UP"),
    "DOM_MIG":      ("DOM_MIG_1S_LOW", "DOM_MIG_1S_UP", "DOM_MIG_2S_LOW", "DOM_MIG_2S_UP"),
    "RESIDUAL":     ("RES_1S_LOW", "RES_1S_UP", "RES_2S_LOW", "RES_2S_UP"),
}

for col, (l1, h1, l2, h2) in plot_config.items():
    plt.figure()

    x = component_projection_model["YEAR"]
    y = component_projection_model[col]

    plt.plot(x, y, label=col)

    # 95% CI (2σ — lighter)
    plt.fill_between(x,
                     component_projection_model[l2],
                     component_projection_model[h2],
                     alpha=0.10, label="95% Confidence Interval")

    # 67% CI (1σ — darker)
    plt.fill_between(x,
                     component_projection_model[l1],
                     component_projection_model[h1],
                     alpha=0.25, label="67% Confidence Interval")

    plt.title(f"{col} Forecast with Confidence Intervals")
    plt.xlabel("Year")
    plt.ylabel(col)
    plt.legend()
    plt.tight_layout()
    plt.show()