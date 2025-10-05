# Computing Flexibility Matrix from a baseline file

import pandas as pd
import numpy as np
import math

# INPUT FILES
BASELINE_XLSX = r"C:\Users\HP\source\repos\Bilevel_Optimization\Bilevel_Optimization\European_LV_CSV\Baseline.xlsx"
BEV_PARAM_XLSX = r"C:\Users\HP\source\repos\Bilevel_Optimization\Bilevel_Optimization\European_LV_CSV\BEV_parameters.xlsx"

# Time step (hours) per row of baseline power
TIME_STEP_H = 1.0

# Baseline xslx has columns: t, charger_id, pbase_kw
baseline = pd.read_excel(BASELINE_XLSX)

# BEV parameters file has: charger_id, eta, p_max, e0
# p_max is maximal charging power and e0 means the initial SoC  upon arrival
bev_parameters = pd.read_excel(BEV_PARAM_XLSX)


required_baseline_cols = {"t", "charger_id", "pbase_kw"}
# quick check
missing = required_baseline_cols - set(baseline.columns)
if missing:
    raise ValueError(f"Baseline is missing columns: {missing}")

required_param_cols = {"charger_id", "eta", "p_max", "e0", "t_dep", "e_target"}
missing_p = required_param_cols - set(bev_parameters.columns)
if missing_p:
    raise ValueError(f"BEV parameters file is missing columns: {missing_p}")

# Ensure types are clean
baseline = baseline.copy()
baseline["t"] = baseline["t"].astype(int)
baseline["pbase_kw"] = pd.to_numeric(baseline["pbase_kw"], errors="coerce").fillna(0.0)



# Parameters
bev_params_map = bev_parameters.set_index("charger_id").to_dict(orient="index")

PARAMS = {}


for b, g in baseline.groupby("charger_id"):
    if b not in bev_params_map:
        raise KeyError(f"charger_id '{b}' found in Baseline but not in BEV_parameters")

    eta  = float(bev_params_map[b]["eta"])  # charging efficiency
    pmax = float(bev_params_map[b]["p_max"]) # maximal charging power
    e0   = float(bev_params_map[b]["e0"])  # initial state of charge 
    t_dep = float(bev_params_map[b]["t_dep"]) # departure time 
    E_target = float(bev_params_map[b]["e_target"]) # target SoC 

    if eta <= 0 or pmax <= 0:
        raise ValueError(f"For charger_id '{b}', eta and p_max must be > 0.")

   

    PARAMS[b] = {
        "E_target": E_target,
        "eta": eta,
        "Pmax": pmax,
        "t_dep": t_dep, 
        "E0": e0,
    }






# --FUNCTIONS --
# Computes SoC by the end of each hour (planned charging)

def compute_soc_end_hour(df: pd.DataFrame, params: dict, time_step_h: float = 1.0) -> pd.DataFrame:
    """
    E_{b,t} = energy at the END of hour t.
      - If pbase_kw == 0 or t > t_dep: Eb_t = 0 , bc the truck left 
      - Else: Eb_t accumulates as E0 + sum P_base up through hour t (no efficiency here, because I want to reach that exact level of Energy by the end of each hour).
    """
    out = df.sort_values(["charger_id", "t"]).reset_index(drop=True).copy()
    out["t_dep_row"] = out["charger_id"].map(lambda b: params[b]["t_dep"])

    Eb = []
    cum_energy = {}

    for _, row in out.iterrows():
        b = row["charger_id"]
        t = int(row["t"])
        p = float(row["pbase_kw"])
        t_dep_b = int(row["t_dep_row"])

        if b not in cum_energy:
            cum_energy[b] = params[b]["E0"]

        # If not charging this hour or after departure: Eb,t = 0 and reset accumulator
        if p == 0 or t > t_dep_b:
            Eb.append(0.0)
            cum_energy[b] = params[b]["E0"]
            continue

        # End-of-hour semantics: add this hour first, then Eb_t reflects end-of-hour t
        cum_energy[b] += p * time_step_h
        Eb.append(cum_energy[b])

    out["Eb_t"] = Eb
    return out.drop(columns=["t_dep_row"])


#Defining the categories for Flexibility Matrix 
def compute_category(t: int, t_dep: int, E_target: float, Eb_t: float,
                     eta: float, Pmax: float, pbase_kw: float) -> int:
    """
    Scalar category for one row:
      - If pbase_kw == 0 or t > t_dep → 0
      - Δt_min = ceil(max(0, (E_target - Eb_t)/(eta*Pmax)))
      - cat    = floor(max(0, t_dep - t - Δt_min))
    """
    if pbase_kw == 0 or t > t_dep:
        return 0
    need_hours = max(0.0, (E_target - Eb_t) / (eta * Pmax))
    dt_min = math.ceil(need_hours)
    cat = max(0, math.floor(t_dep - t - dt_min)) # If truck for example needs 0.4 h to charge here it will be considered as 1 h for the sake of the simplicity.
    return int(cat)




def compute_shiftability_end_hour(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Δt_min = ceil(max(0, (E_target - Eb_t)/(η*Pmax)))
    cat_bt  = floor(max(0, t_dep - t - Δt_min))
    Force 0 if pbase_kw=0 or t > t_dep.
    """
    out = df.copy()
    for key in ("E_target", "Pmax", "eta", "t_dep"):
        out[key] = out["charger_id"].map(lambda b: params[b][key])

    # Needed time (continuous), then round UP
    dt_cont = (out["E_target"] - out["Eb_t"]) / (out["eta"] * out["Pmax"])
    dt_min = np.ceil(np.clip(dt_cont, a_min=0, a_max=None))

    # Absent if no charging now or t beyond departure
    absent = (out["pbase_kw"] == 0) | (out["t"] > out["t_dep"])
    dt_min = np.where(absent, 0, dt_min)
    out["dt_min"] = dt_min.astype(float)

    # Category = t_dep - t - dt_min, floor to int, zero if absent
    cat_cont = (out["t_dep"] - out["t"] - out["dt_min"])
    cat_cont = np.where(absent, 0, np.maximum(0, cat_cont))
    out["cat_bt"] = np.floor(cat_cont).astype(int)

    return out


def build_flex_matrix(df: pd.DataFrame, K=None) -> pd.DataFrame:
    """
    F^{flex}_{t,k} = Σ_b P^{Base}_{b,t} * 1{ cat_{b,t} ≥ k },  k = 0..K
    """
    if K is None:
        K = int(df["cat_bt"].max()) if len(df) else 0

    result = {}
    for k in range(K + 1):
        mask = (df["cat_bt"] >= k).astype(int)
        result[k] = (df["pbase_kw"] * mask).groupby(df["t"]).sum()

    Fflex = pd.DataFrame(result)
    Fflex.columns = [f"k={c}" for c in Fflex.columns]
    Fflex.index.name = "t"
    Fflex.columns.name = "delay bucket"
    return Fflex








#RUN
soc_df = compute_soc_end_hour(baseline, PARAMS, time_step_h=TIME_STEP_H)
shift_df = compute_shiftability_end_hour(soc_df, PARAMS)
Fflex = build_flex_matrix(shift_df)

print("\n--- Shiftability details ---")
print(
    shift_df[["t", "charger_id", "pbase_kw", "Eb_t", "dt_min", "cat_bt"]]
    .round(2)
    .to_string(index=False)
)

print("\n--- Flexibility Matrix ---")
print(Fflex.round(2).to_string())

Fflex.to_csv("flex_matrix.csv", index=True)
print("\nSaved flexibility matrix to flex_matrix.csv")