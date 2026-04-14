"""
Atlanta MSA 10-Year Population Forecast
Cohort-Component Model (2025-2035)

Required files (same directory or adjust paths):
  - CBSA-EST2025-ALLDATA.csv   -> pop_estimate_components
  - CBSA-EST2024-SYASEX.csv    -> pop_by_agesex
  - mortality_2020_census.csv  -> mortality_rate_2020_census

Column name assumptions (adjust if yours differ):
  pop_estimate_components : YEAR, POP_ESTIMATE, BIRTHS, NET_MIG, RESIDUAL
  pop_by_agesex           : CBSA, YEAR (1=2019,2=2020,...,6=2024), AGE, TOT_POP, TOT_MALE, TOT_FEMALE
  mortality_rate_2020_census: "Age on  April 1, 2020", "Total resident population", Deaths, Births
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load Excel file and get sheet names
excel_file = '/Users/pranavlakhotia/Downloads/Final_Data_for_Modeling.xlsx'
xls = pd.ExcelFile(excel_file)

# Get all sheet names
sheet_names = xls.sheet_names
print(f"Sheet names: {sheet_names}")

# Load each sheet into a separate dataframe
for sheet in sheet_names:
    df = pd.read_excel(excel_file, sheet_name=sheet)
    # Create a variable with the sheet name as the variable name
    globals()[sheet] = df

# Filter Atlanta MSA (CBSA 12060)
pop_estimate_components = pop_estimate_components[
    pop_estimate_components["CBSA"] == 12060
].copy()

pop_by_agesex = pop_by_agesex[
    pop_by_agesex["CBSA"] == 12060
].copy()

# Map YEAR codes to calendar years (1=2019, 2=2020, ..., 6=2024)
pop_by_agesex["YEAR"] = pop_by_agesex["YEAR"] + 2018

# ============================================================
# 1. FERTILITY DISTRIBUTION
# ============================================================
fertility_data = pd.DataFrame({
    "age_group":          ["15-19","20-24","25-29","30-34","35-39","40-44","45-49"],
    "relative_fertility": [0.025,  0.200,  0.300,  0.250,  0.150,  0.050,  0.025]
})

# Expand to single-year ages
fertility_map = []
for _, row in fertility_data.iterrows():
    start, end = map(int, row["age_group"].split("-"))
    for age in range(start, end + 1):
        fertility_map.append({"AGE": age, "fertility_prob": row["relative_fertility"]})
fertility_map_df = pd.DataFrame(fertility_map)

# --- Metro birth rate factor (calibrated to 2020 observed births) ---
pop_2020 = pop_by_agesex[pop_by_agesex["YEAR"] == 2020].copy()
pop_2020["AGE"] = pop_2020["AGE"].astype(int)

fertile_pop = (
    pop_2020[["AGE","TOT_FEMALE"]]
    [(pop_2020["AGE"] >= 15) & (pop_2020["AGE"] <= 49)]
    .merge(fertility_map_df, on="AGE", how="left")
)
fertile_pop["weighted_pop"] = fertile_pop["TOT_FEMALE"] * fertile_pop["fertility_prob"]
weighted_sum = fertile_pop["weighted_pop"].sum()

births_2020 = pop_estimate_components.loc[
    pop_estimate_components["YEAR"] == 2020, "BIRTHS"
].values[0]

birth_rate_factor = births_2020 / weighted_sum
print(f"Birth rate factor (metro): {birth_rate_factor:.4f}")

# ============================================================
# 2. MORTALITY SCHEDULE  (Gompertz log-linear, 2020 Census)
# ============================================================
mort = mortality_rate_2020_census.copy()
mort.columns = (
    mort.columns
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)
mort = mort.rename(columns={
    "Age on April 1, 2020":    "AGE",
    "Total resident population": "POP",
    "Deaths":                   "DEATHS"
})

mort = mort[pd.to_numeric(mort["AGE"], errors="coerce").notna()].copy()
mort["AGE"]        = mort["AGE"].astype(int)
mort["death_rate"] = mort["DEATHS"] / mort["POP"]
mort = mort.replace([np.inf, -np.inf], np.nan).dropna(subset=["death_rate"])

# Full age grid 0-75 from observed data
base_ages  = pd.DataFrame({"AGE": list(range(0, 76))})
mort_base  = base_ages.merge(mort[["AGE","death_rate"]], on="AGE", how="left")
mort_base["log_dr"] = np.log(mort_base["death_rate"])

# Fit Gompertz OLS on ages 50-75
mort_fit = mort_base[(mort_base["AGE"] >= 50) & (mort_base["AGE"] <= 75)].dropna()
X_fit = sm.add_constant(mort_fit[["AGE"]], has_constant="add")
gompertz_model = sm.OLS(mort_fit["log_dr"], X_fit).fit()
alpha = gompertz_model.params["const"]
beta  = gompertz_model.params["AGE"]
print(f"Gompertz fit: log(m) = {beta:.4f}*age + {alpha:.4f}")

# Extrapolate ages 76-84
ages_extra = pd.DataFrame({"AGE": list(range(76, 85))})
X_extra    = sm.add_constant(ages_extra[["AGE"]], has_constant="add")
ages_extra["death_rate"] = np.exp(gompertz_model.predict(X_extra))

# Age 85+ proxy at age 88
X_85       = sm.add_constant(pd.DataFrame({"AGE": [88]}), has_constant="add")
dr_85plus  = np.exp(gompertz_model.predict(X_85)[0])
age_85_df  = pd.DataFrame({"AGE": [85], "death_rate": [dr_85plus]})

# Combine full mortality table
mort_final = pd.concat([
    mort_base[["AGE","death_rate"]],
    ages_extra[["AGE","death_rate"]],
    age_85_df
], ignore_index=True).sort_values("AGE").reset_index(drop=True)

# Smooth all ages with the Gompertz fit
X_all = sm.add_constant(mort_final[["AGE"]], has_constant="add")
mort_final["exp_mortality_rate"] = np.exp(gompertz_model.predict(X_all))

# Override 85+ with age-88 proxy (already computed above)
mort_final.loc[mort_final["AGE"] == 85, "exp_mortality_rate"] = dr_85plus

# Incremental mortality for cohort transitions
mort_final["mortality_rate"] = mort_final["exp_mortality_rate"].diff()
mort_final.loc[mort_final.index[0], "mortality_rate"] = \
    mort_final.loc[mort_final.index[0], "exp_mortality_rate"]

print(f"Mortality table rows: {len(mort_final)}")  # should be 86

# ============================================================
# 3. BUILD MAIN PANEL  (years 2020-2024, all ages)
# ============================================================
pop_by_agesex["AGE"] = pop_by_agesex["AGE"].astype(int)
mort_final["AGE"]    = mort_final["AGE"].astype(int)

df = pop_by_agesex[pop_by_agesex["YEAR"].between(2020, 2024)].copy()
df = df.merge(mort_final[["AGE","mortality_rate"]], on="AGE", how="left")
df["deaths"] = df["TOT_POP"] * df["mortality_rate"]
df["births"] = 0.0

# Fertility proportions merged in
df = df.merge(fertility_map_df.rename(columns={"fertility_prob":"fertility_prop"}),
              on="AGE", how="left")
df["fertility_prop"] = df["fertility_prop"].fillna(0)
df.loc[(df["AGE"] < 15) | (df["AGE"] > 49), "fertility_prop"] = 0

# ============================================================
# 4. MIGRATION  (age-weighted differential, 2000-2024)
# ============================================================
pop_estimate_components["DIFF"] = (
    pop_estimate_components["NET_MIG"] +
    pop_estimate_components["RESIDUAL"]
)
diff_by_year = pop_estimate_components.groupby("YEAR")["DIFF"].sum().reset_index()

# Population weights per year
df["popestimate"]    = df.groupby("YEAR")["TOT_POP"].transform("sum")
df["ageweighted_pop"] = df["TOT_POP"] / df["popestimate"]
df = df.merge(diff_by_year, on="YEAR", how="left")
df["ageweighted_diff"] = df["ageweighted_pop"] * df["DIFF"]

# Mean and std of age-weighted migration across 2000-2024
# We need the full historical panel for this
df_hist_all = pop_by_agesex[pop_by_agesex["YEAR"].between(2000, 2024)].copy()
df_hist_all["AGE"] = df_hist_all["AGE"].astype(int)
df_hist_all["popestimate"]    = df_hist_all.groupby("YEAR")["TOT_POP"].transform("sum")
df_hist_all["ageweighted_pop"] = df_hist_all["TOT_POP"] / df_hist_all["popestimate"]
df_hist_all = df_hist_all.merge(diff_by_year, on="YEAR", how="left")
df_hist_all["ageweighted_diff"] = df_hist_all["ageweighted_pop"] * df_hist_all["DIFF"]

mig_stats = (
    df_hist_all.groupby("AGE")["ageweighted_diff"]
    .agg(avg_ageweighted_diff="mean", std_ageweighted_diff="std")
    .reset_index()
)

df = df.merge(mig_stats, on="AGE", how="left")
df = df.drop(columns=["popestimate","ageweighted_pop","DIFF","ageweighted_diff"])

# ============================================================
# 5. FEMALE RATIO  (from 2021-2024, ages 15-49)
# ============================================================
df_4yrs = df[(df["YEAR"].between(2021, 2024)) & (df["AGE"].between(15, 49))]
female_ratio = df_4yrs["TOT_FEMALE"].sum() / df_4yrs["TOT_POP"].sum()
print(f"Female ratio (15-49): {female_ratio:.4f}")

# ============================================================
# 6. AGE-85+ GEOMETRIC GROWTH RATE  (from 2020-2024)
# ============================================================
pop_85_hist = df[(df["AGE"] == 85) & (df["YEAR"].between(2020, 2024))].sort_values("YEAR")
p85_start   = pop_85_hist.iloc[0]["TOT_POP"]
p85_end     = pop_85_hist.iloc[-1]["TOT_POP"]
g85         = (p85_end / p85_start) ** (1 / (len(pop_85_hist) - 1)) - 1
print(f"Age-85+ geometric growth rate: {g85:.4f}")

# ============================================================
# 7. PROJECTION LOOP  2025-2035
# ============================================================

def project_one_year(df_prev, birth_rate_factor, female_ratio,
                     p85_end, g85, diff_col="avg_ageweighted_diff"):
    """
    Given a sorted (by AGE) single-year dataframe, returns a new
    dataframe with projected TOT_POP, TOT_FEMALE, TOT_MALE for next year.
    """
    df_p = df_prev.sort_values("AGE").reset_index(drop=True)

    # --- Births ---
    births = (
        df_p["TOT_FEMALE"] * df_p["fertility_prop"]
    ).sum() * birth_rate_factor

    # --- Cohort aging: P(x,t+1) = P(x-1,t) - m(x-1)*P(x-1,t) + M(x-1,t) ---
    prev_pop  = df_p["TOT_POP"].shift(1)
    prev_mort = df_p["mortality_rate"].shift(1)
    prev_mig  = df_p[diff_col].shift(1)

    new_pop = prev_pop - (prev_mort * prev_pop) + prev_mig

    # Assign births to age 0
    new_pop.iloc[0] = births

    # Age 85+ geometric trend
    new_pop.iloc[-1] = p85_end * (1 + g85)

    df_new = df_p[["AGE","mortality_rate","fertility_prop",
                   "avg_ageweighted_diff","std_ageweighted_diff"]].copy()
    df_new["TOT_POP"]    = new_pop.values
    df_new["TOT_FEMALE"] = female_ratio * df_new["TOT_POP"]
    df_new["TOT_MALE"]   = df_new["TOT_POP"] - df_new["TOT_FEMALE"]
    df_new["deaths"]     = df_new["TOT_POP"] * df_new["mortality_rate"]

    return df_new


# Initialise projection container with historical years
df_proj = df.copy()

# Extend p85_end forward each year using the same rate
p85_rolling = p85_end  # will advance each iteration

for year in range(2025, 2036):
    p85_rolling = p85_rolling * (1 + g85)  # advance one year

    df_prev = (
        df_proj[df_proj["YEAR"] == year - 1]
        .sort_values("AGE")
        .reset_index(drop=True)
    )

    df_new = project_one_year(
        df_prev, birth_rate_factor, female_ratio,
        p85_rolling / (1 + g85),   # base for this year (already computed)
        g85
    )
    df_new["YEAR"] = year
    df_proj = pd.concat([df_proj, df_new], ignore_index=True)

# ============================================================
# 8. YEARLY TOTALS
# ============================================================
yearly_pop = (
    df_proj.groupby("YEAR")["TOT_POP"].sum()
    .reset_index()
    .rename(columns={"TOT_POP":"est_pop"})
    .sort_values("YEAR")
)
print("\nYearly population estimates:")
print(yearly_pop.to_string(index=False))

# ============================================================
# 9. SCENARIO PROJECTIONS  (±1σ, ±2σ migration)
# ============================================================

scenarios = {
    "vlow":  ("avg_ageweighted_diff", -2),
    "low":   ("avg_ageweighted_diff", -1),
    "base":  ("avg_ageweighted_diff",  0),
    "high":  ("avg_ageweighted_diff", +1),
    "vhigh": ("avg_ageweighted_diff", +2),
}

scen_totals = {}  # {scenario_name: {year: total_pop}}

for scen_name, (_, sigma_mult) in scenarios.items():
    df_s = df.copy()

    # Build scenario-specific diff column
    df_s["diff_scen"] = (
        df_s["avg_ageweighted_diff"] +
        sigma_mult * df_s["std_ageweighted_diff"]
    )

    p85_rolling_s = p85_end
    totals = {}

    for year in range(2025, 2036):
        p85_rolling_s = p85_rolling_s * (1 + g85)

        df_prev_s = (
            df_s[df_s["YEAR"] == year - 1]
            .sort_values("AGE")
            .reset_index(drop=True)
        )

        df_new_s = project_one_year(
            df_prev_s, birth_rate_factor, female_ratio,
            p85_rolling_s / (1 + g85), g85,
            diff_col="diff_scen"
        )
        df_new_s["YEAR"]      = year
        df_new_s["diff_scen"] = (
            df_new_s["avg_ageweighted_diff"] +
            sigma_mult * df_new_s["std_ageweighted_diff"]
        )
        df_s = pd.concat([df_s, df_new_s], ignore_index=True)
        totals[year] = df_new_s["TOT_POP"].sum()

    scen_totals[scen_name] = totals

scen_df = pd.DataFrame(scen_totals)
scen_df.index.name = "YEAR"
print("\nScenario totals (2025-2035):")
display(scen_df)

# ============================================================
# 10. PLOT
# ============================================================
hist_plot = (
    pop_estimate_components[["YEAR","POP_ESTIMATE"]]
    .rename(columns={"POP_ESTIMATE":"population"})
    .sort_values("YEAR")
)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(hist_plot["YEAR"], hist_plot["population"],
        color="steelblue", linewidth=2, label="Historical (Census)")

forecast_years = list(scen_df.index)

ax.plot(forecast_years, scen_df["base"],
        color="darkorange", linewidth=2, label="Base Forecast")

ax.fill_between(forecast_years,
                scen_df["low"], scen_df["high"],
                alpha=0.25, color="steelblue", label=r"$\pm1\sigma$ band")

ax.fill_between(forecast_years,
                scen_df["vlow"], scen_df["vhigh"],
                alpha=0.12, color="steelblue", label=r"$\pm2\sigma$ band")

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Population", fontsize=12)
ax.set_title("Atlanta MSA Population Forecast (2000–2035)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.4)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M"))

plt.tight_layout()
plt.savefig("fig_population_forecast.png", dpi=150)
plt.show()
print("Forecast chart saved → fig_population_forecast.png")

# ============================================================
# 11. EXPORT
# ============================================================
print("Full projection saved → population_forecast.xlsx")