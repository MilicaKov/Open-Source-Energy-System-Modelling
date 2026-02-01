# ======================================================================
# SECTION 0 — IMPORTS, SWITCHES, FILE PATHS, CONSTANTS
# ======================================================================
import pyomo.environ as pyo
import pandas as pd
from collections import defaultdict
import numpy as np
from pyomo.environ import value
import matplotlib.pyplot as plt
import math
import warnings

USE_KKT     = True
USE_NETWORK = True

BEV_XLSX        = r"C:\Users\HP\Desktop\Code\Inputs\BEV_parameters.xlsx"
TIMESERIES_XLSX = r"C:\Users\HP\Desktop\Code\Inputs\Time_Series.xlsx"

DELTA_T = 1.0

# tariff bounds
LAMBDA_MIN = 200
LAMBDA_MAX = 1200
DELTA_LAMBDA_MAX = 10

# tariff ladder used in your model (keep one definition only)
LAMBDA_STEPS = {0: 0.0, 1: 200.0, 2: 500.0, 3: 800.0, 4: 1200.0}

# congestion regime thresholds (used ONLY for c_dict)
THETA_REGIME = {1: 1.0, 2: 5.0}  # <=1e-6 -> normal; <THETA_REGIME[2] -> critical; else congested

DELTA_P_MAX = 1.5

C_VOLT   = 3000
C_THERM  = 6000
C_ENS    = 3000000
C_PVCURT = 950

C_SHORT  = 800
W_EPS    = 1e5

P_CONN = 4
V_E_CARRY  = 0

V_MIN = 0.90**2
V_MAX = 1.05**2


TARGET_MIN_FRAC = 0.7
SOC_MIN_DEP     = 0.80

# ======================================================================
# SECTION 1 — READ INPUTS + BUILD ALL DATA DICTS/SETS (NO PYOMO YET)
# ======================================================================
bev      = pd.read_excel(BEV_XLSX)


bev["charger_id"]      = bev["charger_id"].astype(str)
T = list(range(24))   # IEEE 33 daily horizon
H = len(T)
print("Hours:", T)

def parse_hour(x):
    if isinstance(x, (int, float, np.integer, np.floating)):
        return int(x)
    x = str(x).strip()
    return int(x.split(":")[0])


def availability(ta, td, t):
    if ta < td:
        return 1 if (ta <= t < td) else 0
    else:
        return 1 if (t >= ta or t < td) else 0

t_arr_map, t_dep_map = {}, {}
for _, row in bev.iterrows():
    b = str(row["charger_id"])
    t_arr_map[b] = int(row["t_arr"]) % 24
    t_dep_map[b] = int(row["t_dep"]) % 24

t_last_map = {b: (t_dep_map[b] - 1) % 24 for b in t_dep_map.keys()}

# --- load / PV / network ---
df_load = pd.read_excel(TIMESERIES_XLSX, sheet_name="load_profile")
df_load.rename(columns={df_load.columns[0]: "time_str"}, inplace=True)
df_load = df_load[df_load["time_str"].notna()]




df_Qload = pd.read_excel(TIMESERIES_XLSX, sheet_name="load_Q")
df_Qload.rename(columns={df_Qload.columns[0]: "time_str"}, inplace=True)
df_Qload["t"] = df_Qload["time_str"].apply(parse_hour).astype(int)



#replace NaNs with 0
df_load = df_load.fillna(0.0)

def parse_hour(x):
    if isinstance(x, (int, float, np.integer, np.floating)):
        return int(x)
    x = str(x).strip()
    return int(x.split(":")[0])

df_load["t"] = df_load["time_str"].apply(parse_hour).astype(int)
bus_cols = [c for c in df_load.columns if c not in ["time_str", "t"]]


df_load_long = df_load.melt(
    id_vars=["t"], value_vars=bus_cols, var_name="bus", value_name="P_load_pu"
)
df_load_long["bus"] = df_load_long["bus"].astype(int)
df_load_long["t"]   = df_load_long["t"].astype(int)

df_pv = pd.read_excel(TIMESERIES_XLSX, sheet_name="PV_profile")
if "bus_id" in df_pv.columns:
    df_pv.rename(columns={"bus_id": "bus"}, inplace=True)
df_pv["bus"] = df_pv["bus"].astype(int)
df_pv["t"]   = df_pv["t"].astype(int)

net = pd.read_excel(TIMESERIES_XLSX, sheet_name="network")
net["line_id"]  = net["line_id"].astype(int)
net["from_bus"] = net["from_bus"].astype(int)
net["to_bus"]   = net["to_bus"].astype(int)

slack_bus = int(pd.read_excel(TIMESERIES_XLSX, sheet_name="slack_bus").iloc[0, 0])
df_fleet = pd.read_excel(TIMESERIES_XLSX, sheet_name="fleet_buses")
fleet_buses = list(df_fleet["bus"].astype(int))
P_conn_map = dict(zip(df_fleet["bus"], df_fleet["P_conn"]))


I        = sorted(set(net["from_bus"]).union(set(net["to_bus"])))
line_ids = list(net["line_id"])

from_bus_map = {int(r["line_id"]): int(r["from_bus"]) for _, r in net.iterrows()}
to_bus_map   = {int(r["line_id"]): int(r["to_bus"])   for _, r in net.iterrows()}
R_map        = {int(r["line_id"]): float(r["R"])      for _, r in net.iterrows()}
X_map        = {int(r["line_id"]): float(r["X"])      for _, r in net.iterrows()}
Smax_map     = {int(r["line_id"]): float(r["S_max"])  for _, r in net.iterrows()}

P_load_dict = {(i, t): 0.0 for i in I for t in T}
PV_dict     = {(i, t): 0.0 for i in I for t in T}

for _, r in df_load_long.iterrows():
    i = int(r["bus"]); t = int(r["t"])
    if i in I and t in T:
        P_load_dict[(i, t)] = float(r["P_load_pu"])

for _, r in df_pv.iterrows():
    i = int(r["bus"]); t = int(r["t"])
    if i in I and t in T:
        PV_dict[(i, t)] = float(r["PV"])


Qload_dict = {(i,t): 0.0 for i in I for t in T}

for _, r in df_Qload.iterrows():
    t = int(r["t"])
    if t in T:
        for bus in df_Qload.columns:
            if bus not in ["time_str", "t"]:
                i = int(bus)
                Qload_dict[(i,t)] = float(r[bus])

# --- price (OK if missing; stays 0) ---
price_dict = {t: 0.0 for t in T}
try:
    prices_raw = pd.read_excel(TIMESERIES_XLSX, sheet_name="price_profile")
    if "t" not in prices_raw.columns:
        prices_raw.rename(columns={prices_raw.columns[0]: "time_str"}, inplace=True)
        prices_raw["t"] = prices_raw["time_str"].apply(lambda x: int(str(x).split(":")[0]))
    prices_raw["t"] = prices_raw["t"].astype(int)
    prices_raw = prices_raw[prices_raw["t"].isin(T)]
    for _, row in prices_raw.iterrows():
        price_dict[int(row["t"])] = float(row["price"])
except Exception:
    pass

# --- aggregate fleet ---
B = list(bev["charger_id"].unique())

def bev_scalar(b, col):
    return float(bev.loc[bev["charger_id"] == b, col].iloc[0])

Pmax_fleet = 1.3 * sum(bev.loc[bev["charger_id"] == b, "p_max"].iloc[0] for b in B)
Emax_fleet = sum(bev_scalar(b, "battery_size") for b in B)
E0_fleet   = sum(bev_scalar(b, "e0")       * bev_scalar(b, "battery_size") for b in B)
Etarget_fleet = sum(bev_scalar(b, "e_target") * bev_scalar(b, "battery_size") for b in B)
eta_fleet  = float(bev["eta"].mean())

a_fleet = {t: int(any(availability(t_arr_map[b], t_dep_map[b], t) for b in B)) for t in T}
t_last_fleet = max((t_dep_map[b] - 1) % 24 for b in B)

def downstream_buses(start_bus):
    visited = set()
    stack = [start_bus]
    while stack:
        b = stack.pop()
        for _, r in net.iterrows():
            if r["from_bus"] == b and r["to_bus"] not in visited:
                visited.add(r["to_bus"])
                stack.append(r["to_bus"])
    return visited


#Before Optimization

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------
# Aggregate gross load (input only)
# ---------------------------------
T_list = list(T)

total_load_input = np.array([
    sum(P_load_dict[(i, t)] for i in I)
    for t in T_list
])


# ======================================================================
# SECTION 2 — MODEL BUILDER (BUILD ONCE, USE TWICE)
#   - baseline: enforce_regime=False
#   - tariff:   enforce_regime=True + c_dict passed
# ======================================================================
def build_model(c_init=None, enforce_regime=True, sigma_base_init=None, sigma_fc_init=None, include_network=True):

    if c_init is None:
        c_init = {t: 0 for t in T}

    if sigma_fc_init is None:
        sigma_fc_init = {t: 0.0 for t in T}

    m = pyo.ConcreteModel()

    # sets
    m.T = pyo.Set(initialize=T, ordered=True)

    # params
    m.sigma_fc = pyo.Param(m.T, initialize=sigma_fc_init, mutable=False)

    # ======================================================
    # SETS & EXOGENOUS PARAMETERS
    # ======================================================
    

    m.c = pyo.Param(
        m.T,
        initialize=c_init,
        within=pyo.NonNegativeIntegers,
        mutable=False
    )

    m.price = pyo.Param(m.T, initialize=price_dict, mutable=False)

    # ======================================================
    # UPPER LEVEL — TARIFF
    # ======================================================
    m.lambda_t = pyo.Var(m.T, bounds=(0.0, LAMBDA_MAX))

    m.K = pyo.Set(initialize=sorted(LAMBDA_STEPS.keys()))
    m.z_step = pyo.Var(m.K, m.T, domain=pyo.Binary)

    m.ONE_STEP = pyo.Constraint(
        m.T, rule=lambda m, t: sum(m.z_step[k, t] for k in m.K) == 1
    )

    m.LAMBDA_DEF = pyo.Constraint(
        m.T,
        rule=lambda m, t:
            m.lambda_t[t]
            == sum(LAMBDA_STEPS[k] * m.z_step[k, t] for k in m.K)
    )

    EPS_CONG = 1e-6
    THETA1 = THETA_REGIME[1]    # boundary normal/critical
    THETA2 = THETA_REGIME[2]    # boundary critical/congested


    K_N = [0, 1]      # normal: λ = 0 or 60
    K_C = [2]         # critical: λ = 120
    K_G = [3, 4]      # congested: λ = 180 or 240

    def activation_rule(m, t):
        s = float(m.sigma_fc[t])

        if s <= EPS_CONG:
            return sum(m.z_step[k, t] for k in K_N) == 1
        elif s < THETA2:
            return sum(m.z_step[k, t] for k in K_C) == 1
        else:
            return sum(m.z_step[k, t] for k in K_G) == 1

    # ✅ Only enforce regime logic when requested
    if enforce_regime:
        m.ACTIVATION = pyo.Constraint(m.T, rule=activation_rule)



    # ======================================================
    # LOWER LEVEL — FLEET
    # ======================================================
    m.P      = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.E      = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.Pfleet = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    m.s_dep   = pyo.Var(domain=pyo.NonNegativeReals)
    m.SoC_dep = pyo.Var(bounds=(0.0, 1.0))

    m.a = pyo.Param(m.T, initialize=a_fleet, within=pyo.Binary)

    m.P_LIMIT = pyo.Constraint(
        m.T, rule=lambda m, t: m.P[t] <= m.a[t] * Pmax_fleet
    )


    def soc_rule(m, t):
        if t == m.T.first():
            return m.E[t] == E0_fleet
        return m.E[t] == m.E[m.T.prev(t)] + eta_fleet * DELTA_T * m.P[t]

    m.SOC = pyo.Constraint(m.T, rule=soc_rule)

    m.E_MAX = pyo.Constraint(
        m.T, rule=lambda m, t: m.E[t] <= Emax_fleet
    )

    m.PFLEET_LINK = pyo.Constraint(
        m.T, rule=lambda m, t: m.Pfleet[t] == m.P[t]
    )


    if Emax_fleet > 0:
        m.SOC_DEP_DEF = pyo.Constraint(expr=m.SoC_dep == m.E[t_last_fleet] / Emax_fleet)
    else:
        pass



    m.SOC_MIN_DEP = pyo.Constraint(
        expr=m.E[t_last_fleet] >= SOC_MIN_DEP * Emax_fleet
    )

    m.SOFT_TARGET = pyo.Constraint(
        expr=m.E[t_last_fleet] + m.s_dep >= Etarget_fleet
    )

    # ======================================================
    # NETWORK
    # ======================================================
    if include_network and USE_NETWORK:

        m.I = pyo.Set(initialize=I)
        m.L = pyo.Set(initialize=line_ids)
        m.FB = pyo.Set(initialize=fleet_buses)   # e.g. [6,18,25,30]


        m.Pload = pyo.Param(m.I, m.T, initialize=P_load_dict)
        m.Qload = pyo.Param(m.I, m.T, initialize=Qload_dict, mutable=False)

        m.PPV   = pyo.Param(m.I, m.T, initialize=PV_dict)

        m.R    = pyo.Param(m.L, initialize=R_map)
        m.X    = pyo.Param(m.L, initialize=X_map)
        m.Smax = pyo.Param(m.L, initialize=Smax_map)


        m.Pij = pyo.Var(m.L, m.T)
        m.Qij = pyo.Var(m.L, m.T)
        m.V   = pyo.Var(m.I, m.T)
        m.Qslack = pyo.Var(m.T)
        m.Pslack = pyo.Var(m.T)


        m.ENS    = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)
        m.PVcurt = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)

        m.xi_low  = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)
        m.xi_high = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)

        m.sigma = pyo.Var(m.L, m.T, domain=pyo.NonNegativeReals)

        m.QPV = pyo.Var(m.I, m.T)         # PV inverter reactive power


        m.Pfleet_bus = pyo.Var(m.FB, m.T, domain=pyo.NonNegativeReals)
        m.Qfleet_bus = pyo.Var(m.FB, m.T)   # can be positive or negative





        m.FLEET_POWER_SPLIT = pyo.Constraint(
            m.T, rule=lambda m, t:
                sum(m.Pfleet_bus[b, t] for b in m.FB) == m.Pfleet[t]
        )

        m.FLEET_BUS_LIMIT = pyo.Constraint(
            m.FB, m.T,
            rule=lambda m, b, t: m.Pfleet_bus[b, t] <= P_conn_map[b]
        )





        m.V_LOW = pyo.Constraint(
            m.I, m.T, rule=lambda m, i, t: m.V[i, t] >= V_MIN - m.xi_low[i, t]
        )
        m.V_HIGH = pyo.Constraint(
            m.I, m.T, rule=lambda m, i, t: m.V[i, t] <= V_MAX + m.xi_high[i, t]
        )

        m.V_SLACK = pyo.Constraint(
            m.T, rule=lambda m, t: m.V[slack_bus, t] == 1.0
        )

        #for i in I:
        #    if i != slack_bus and not any(to_bus_map[l] == i for l in line_ids):
        #       for t in T:
        #            m.V[i, t].fix(1.0)

        #m.V_FIX = pyo.Constraint(
        #    m.I, m.T, 
        #    rule=lambda m,i,t: m.V[i,t] == 1.0
        #    if (i == slack_bus) else pyo.Constraint.Skip
        #)


        def p_balance(m, i, t):
            inflow  = sum(m.Pij[l, t] for l in m.L if to_bus_map[l] == i)
            outflow = sum(m.Pij[l, t] for l in m.L if from_bus_map[l] == i)
            P_fleet = m.Pfleet_bus[i, t] if i in m.FB else 0.0
            P_slack = m.Pslack[t] if i == slack_bus else 0.0
            
            return (
                inflow - outflow
                + m.PPV[i, t]
                - m.Pload[i, t]
                - P_fleet
                + m.ENS[i, t]
                - m.PVcurt[i, t]
                + P_slack
                == 0
            )

        m.P_BAL = pyo.Constraint(m.I, m.T, rule=p_balance)

        def q_balance_rule(m, i, t):
            inflow  = sum(m.Qij[l,t] for l in m.L if to_bus_map[l] == i)
            outflow = sum(m.Qij[l,t] for l in m.L if from_bus_map[l] == i)
            pv_q    = m.QPV[i,t]
            load_q  = m.Qload[i,t]
            fleet_q = m.Qfleet_bus[i,t] if i in m.FB else 0.0
            slack_q = m.Qslack[t] if i == slack_bus else 0.0

            return inflow - outflow + pv_q - load_q - fleet_q + slack_q == 0

        m.Q_BAL = pyo.Constraint(m.I, m.T, rule=q_balance_rule)


        def voltage_drop_rule(m, l, t):
            f = from_bus_map[l]
            g = to_bus_map[l]
            return m.V[g,t] == m.V[f,t] - 2*(m.R[l]*m.Pij[l,t] + m.X[l]*m.Qij[l,t])
        m.VOLT_DROP = pyo.Constraint(m.L, m.T, rule=voltage_drop_rule)


        PF_MIN = 0.95
        TANPHI = np.tan(np.arccos(PF_MIN))

        def qpv_cap_pos(m,i,t):
            return m.QPV[i,t] <= TANPHI*(m.PPV[i,t] - m.PVcurt[i,t])

        def qpv_cap_neg(m,i,t):
            return -m.QPV[i,t] <= TANPHI*(m.PPV[i,t] - m.PVcurt[i,t])

        m.QPV_CAP1 = pyo.Constraint(m.I,m.T, rule=qpv_cap_pos)
        m.QPV_CAP2 = pyo.Constraint(m.I,m.T, rule=qpv_cap_neg)



        PF_FLEET = 0.95 #Passive chargers 
        TANPHI_FLEET = np.tan(np.arccos(PF_FLEET))

        def qfleet_cap_pos(m,t):
            return m.Qfleet[t] <= TANPHI_FLEET * m.Pfleet[t]

        def qfleet_cap_neg(m,t):
            return -m.Qfleet[t] <= TANPHI_FLEET * m.Pfleet[t]

    

        m.QFLEET_CAP_POS = pyo.Constraint(
            m.FB, m.T,
            rule=lambda m,b,t: m.Qfleet_bus[b,t] <= TANPHI_FLEET * m.Pfleet_bus[b,t]
        )

        m.QFLEET_CAP_NEG = pyo.Constraint(
            m.FB, m.T, 
            rule=lambda m,b,t: -m.Qfleet_bus[b,t] <= TANPHI_FLEET * m.Pfleet_bus[b,t]
        )


  




        #m.V_DROP = pyo.Constraint(
        #    m.L, m.T,
        #    rule=lambda m, l, t:
        #        m.V[to_bus_map[l], t]
        #        == m.V[from_bus_map[l], t]
        #        - 2 * (m.R[l] * m.Pij[l, t] + m.X[l] * m.Qij[l, t])
        #)

        SQRT2 = math.sqrt(2.0)
        m.THERM_P_POS = pyo.Constraint(
            m.L, m.T, rule=lambda m, l, t:  m.Pij[l, t] <= m.Smax[l] + m.sigma[l, t]
        )
        m.THERM_P_NEG = pyo.Constraint(
            m.L, m.T, rule=lambda m, l, t: -m.Pij[l, t] <= m.Smax[l] + m.sigma[l, t]
        )
        m.THERM_PQ_POS = pyo.Constraint(
            m.L, m.T, rule=lambda m, l, t:
                m.Pij[l, t] + m.Qij[l, t] <= SQRT2 * m.Smax[l] + m.sigma[l, t]
        )
        m.THERM_PQ_NEG = pyo.Constraint(
            m.L, m.T, rule=lambda m, l, t:
                -m.Pij[l, t] - m.Qij[l, t] <= SQRT2 * m.Smax[l] + m.sigma[l, t]
        )
        m.THERM_Q_POS = pyo.Constraint(
            m.L, m.T, rule=lambda m,l,t:  m.Qij[l,t] <= m.Smax[l] + m.sigma[l,t]
        )
        m.THERM_Q_NEG = pyo.Constraint(
            m.L, m.T, rule=lambda m,l,t: -m.Qij[l,t] <= m.Smax[l] + m.sigma[l,t]
        )


        

    # ======================================================
    # LOWER-LEVEL OBJECTIVE (INACTIVE)
    # ======================================================
    m.LL_OBJ = pyo.Objective(
        expr=
            sum((m.price[t] + m.lambda_t[t]) * m.P[t] * DELTA_T for t in m.T)
            + C_SHORT * m.s_dep
            - V_E_CARRY * m.E[t_last_fleet],
        sense=pyo.minimize
    )
    m.LL_OBJ.deactivate()

    # ======================================================
    # KKT EMBEDDING — COMPLETE & CORRECT
    # ======================================================
    if USE_KKT:

        M_PRIMAL_P = float(Pmax_fleet)
        M_PRIMAL_E = float(Emax_fleet)
        M_DUAL     = 1e4
        M_SHORT    = float(Etarget_fleet)

        m.muP_min = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.muP_max = pyo.Var(m.T, domain=pyo.NonNegativeReals)

        m.muE_min = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.muE_max = pyo.Var(m.T, domain=pyo.NonNegativeReals)

        m.nu = pyo.Var(m.T, domain=pyo.Reals)

        m.gamma     = pyo.Var(domain=pyo.NonNegativeReals)
        m.muSOC_min = pyo.Var(domain=pyo.NonNegativeReals)

        m.muS_min = pyo.Var(domain=pyo.NonNegativeReals)

        m.epsP = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.epsE = pyo.Var(m.T, domain=pyo.NonNegativeReals)

        m.zP_min  = pyo.Var(m.T, domain=pyo.Binary)
        m.zP_max  = pyo.Var(m.T, domain=pyo.Binary)
        m.zE_min  = pyo.Var(m.T, domain=pyo.Binary)
        m.zE_max  = pyo.Var(m.T, domain=pyo.Binary)

        m.zSOCmin = pyo.Var(domain=pyo.Binary)
        m.zSHORT  = pyo.Var(domain=pyo.Binary)
        m.zS_min  = pyo.Var(domain=pyo.Binary)

        # ---- Complementarity constraints (unchanged) ----
        m.PMIN_PRIM = pyo.Constraint(
            m.T, rule=lambda m, t:
                -m.P[t] >= -M_PRIMAL_P * (1 - m.zP_min[t])
        )
        m.PMIN_DUAL = pyo.Constraint(
            m.T, rule=lambda m, t:
                m.muP_min[t] <= M_DUAL * m.zP_min[t]
        )

        m.PMAX_PRIM = pyo.Constraint(
            m.T, rule=lambda m, t:
                (m.P[t] - m.a[t] * Pmax_fleet)
                >= -M_PRIMAL_P * (1 - m.zP_max[t])
        )
        m.PMAX_DUAL = pyo.Constraint(
            m.T, rule=lambda m, t:
                m.muP_max[t] <= M_DUAL * m.zP_max[t]
        )

        m.EMIN_PRIM = pyo.Constraint(
            m.T, rule=lambda m, t:
                -m.E[t] >= -M_PRIMAL_E * (1 - m.zE_min[t])
        )
        m.EMIN_DUAL = pyo.Constraint(
            m.T, rule=lambda m, t:
                m.muE_min[t] <= M_DUAL * m.zE_min[t]
        )

        m.EMAX_PRIM = pyo.Constraint(
            m.T, rule=lambda m, t:
                (m.E[t] - Emax_fleet)
                >= -M_PRIMAL_E * (1 - m.zE_max[t])
        )
        m.EMAX_DUAL = pyo.Constraint(
            m.T, rule=lambda m, t:
                m.muE_max[t] <= M_DUAL * m.zE_max[t]
        )

        Emin_dep = SOC_MIN_DEP * Emax_fleet

        m.SOCMIN_PRIM = pyo.Constraint(
            expr=(Emin_dep - m.E[t_last_fleet])
                 >= -M_PRIMAL_E * (1 - m.zSOCmin)
        )
        m.SOCMIN_DUAL = pyo.Constraint(
            expr=m.muSOC_min <= M_DUAL * m.zSOCmin
        )

        m.SHORT_PRIM = pyo.Constraint(
            expr=(m.E[t_last_fleet] + m.s_dep - Etarget_fleet) >= 0
        )
        m.SHORT_COMP1 = pyo.Constraint(
            expr=(m.E[t_last_fleet] + m.s_dep - Etarget_fleet)
                 <= M_SHORT * (1 - m.zSHORT)
        )
        m.SHORT_COMP2 = pyo.Constraint(
            expr=m.gamma <= M_DUAL * m.zSHORT
        )

        m.SMIN_PRIM = pyo.Constraint(
            expr=-m.s_dep >= -M_SHORT * (1 - m.zS_min)
        )
        m.SMIN_DUAL = pyo.Constraint(
            expr=m.muS_min <= M_DUAL * m.zS_min
        )

        m.ST_S = pyo.Constraint(
            expr=C_SHORT - m.gamma - m.muS_min == 0
        )

        def st_p_expr(m, t):
            if t == m.T.first():
                return (
                    (m.price[t] + m.lambda_t[t]) * DELTA_T
                    + m.muP_max[t] - m.muP_min[t]
                )
            return (
                (m.price[t] + m.lambda_t[t]) * DELTA_T
                - eta_fleet * DELTA_T * m.nu[t]
                + m.muP_max[t] - m.muP_min[t]
            )

        m.ST_P_LOW = pyo.Constraint(
            m.T, rule=lambda m, t: st_p_expr(m, t) >= -m.epsP[t]
        )
        m.ST_P_HIGH = pyo.Constraint(
            m.T, rule=lambda m, t: st_p_expr(m, t) <=  m.epsP[t]
        )

        def st_e_expr(m, t):
            nu_next = 0.0 if t == m.T.last() else m.nu[m.T.next(t)]
            nu_t = 0.0 if t == m.T.first() else m.nu[t]

            terminal = 0.0
            if t == t_last_fleet:
                terminal += m.gamma
                terminal -= m.muSOC_min
                terminal -= V_E_CARRY

            return (nu_t - nu_next) + m.muE_max[t] - m.muE_min[t] + terminal

        m.ST_E_LOW = pyo.Constraint(
            m.T, rule=lambda m, t: st_e_expr(m, t) >= -m.epsE[t]
        )
        m.ST_E_HIGH = pyo.Constraint(
            m.T, rule=lambda m, t: st_e_expr(m, t) <=  m.epsE[t]
        )


    # ======================================================
    # UPPER-LEVEL OBJECTIVE
    # ======================================================
    if include_network and USE_NETWORK:
        net_cost = (
            sum(C_VOLT * (m.xi_low[i, t] + m.xi_high[i, t]) for i in m.I for t in m.T)
            + sum(C_THERM * m.sigma[l, t] for l in m.L for t in m.T)
            + sum(C_ENS * m.ENS[i, t] for i in m.I for t in m.T)
            + sum(C_PVCURT * m.PVcurt[i, t] for i in m.I for t in m.T)
        )
    else:
        net_cost = 0.0

    kkt_penalty = W_EPS * sum(m.epsP[t] + m.epsE[t] for t in m.T) if USE_KKT else 0.0

    m.OBJ = pyo.Objective(expr=net_cost + kkt_penalty, sense=pyo.minimize)


    return m



# ======================================================================
# SECTION 3 — SOLVER
# ======================================================================
solver = pyo.SolverFactory("gurobi")

# ======================================================================
# SECTION 4 — BASELINE SOLVE (NO TARIFF, FLEET OPTIMAL RESPONSE)
# ======================================================================



m_base = build_model(
    c_init={t: 0 for t in T},
    enforce_regime=False,
    sigma_fc_init={t: 0.0 for t in T}, 
    include_network=False 
)

# --------------------------------------------------
# 1. Force tariff to zero
# --------------------------------------------------

for t in m_base.T:
    m_base.lambda_t[t].fix(0.0)
for k in m_base.K:
    for t in m_base.T:
        m_base.z_step[k, t].fix(1 if k == 0 else 0)

m_base.LL_OBJ.activate()
m_base.OBJ.deactivate()



# --------------------------------------------------
# 2. ACTIVATE fleet objective
# --------------------------------------------------
m_base.LL_OBJ.activate()

# --------------------------------------------------
# 3. DEACTIVATE upper-level objective
#    (baseline should NOT minimize grid penalties)
# --------------------------------------------------
m_base.OBJ.deactivate()

# --------------------------------------------------
# 4. KEEP SOC / departure constraints ON
#    (this forces nonzero charging)
# --------------------------------------------------
# Do NOT deactivate:
# - SOC
# - SOC_MIN_DEP
# - SOFT_TARGET
# - SOC_DEP_DEF

# --------------------------------------------------
# 5. DEACTIVATE KKT stationarity & complementarity
#    (this is now a single-level LP)
# --------------------------------------------------
for comp in [
    "PMIN_PRIM","PMIN_DUAL","PMAX_PRIM","PMAX_DUAL",
    "EMIN_PRIM","EMIN_DUAL","EMAX_PRIM","EMAX_DUAL",
    "SOCMIN_PRIM","SOCMIN_DUAL",
    "ST_P_LOW","ST_P_HIGH","ST_E_LOW","ST_E_HIGH",
    "SHORT_PRIM","SHORT_COMP1","SHORT_COMP2",
    "SMIN_PRIM","SMIN_DUAL","ST_S",
]:
    if hasattr(m_base, comp):
        getattr(m_base, comp).deactivate()



# --------------------------------------------------
# 6. Solve baseline
# --------------------------------------------------
res_base = solver.solve(m_base, tee=True)

term_base = res_base.solver.termination_condition
print("Baseline termination:", term_base)

if term_base not in (
    pyo.TerminationCondition.optimal,
    pyo.TerminationCondition.feasible,
):
    raise RuntimeError(f"Baseline infeasible: {term_base}")

# --------------------------------------------------
# 7. Extract baseline fleet charging
# --------------------------------------------------
Pfleet_baseline = np.array([
    pyo.value(m_base.Pfleet[t]) for t in T
])


Pfleet_notariff = np.array([pyo.value(m_base.Pfleet[t]) for t in T])
Pfleet_baseline = Pfleet_notariff.copy()

# ======================================================================
# SECTION 5 — FORECAST σ_fc USING NETWORK POWER FLOW (CONSISTENT PHYSICS)
# ======================================================================

def compute_sigma_fc_passive(Pfleet_fc_schedule):

    sigma_fc_line = {(l,t): 0.0 for l in line_ids for t in T}
    sigma_fc_time = {t: 0.0 for t in T}

    # Precompute downstream sets
    downstream = {i: downstream_buses(i) for i in I}

    for k, t in enumerate(T):

        # --- Net injections ---
        Pinj = {}
        for i in I:
            fleet = Pfleet_fc_schedule[k] if i in fleet_buses else 0.0
            Pinj[i] = PV_dict[(i,t)] - P_load_dict[(i,t)] - fleet

        # --- Branch flows (radial sum) ---
        Pij = {}
        for l in line_ids:
            to_bus = to_bus_map[l]
            Pij[l] = sum(-Pinj[j] for j in downstream[to_bus] | {to_bus})

        # --- Voltages ---
        V = {slack_bus: 1.0}
        visited = {slack_bus}

        changed = True
        while changed:
            changed = False
            for l in line_ids:
                f = from_bus_map[l]
                g = to_bus_map[l]
                if f in V and g not in V:
                    V[g] = V[f] - 2*(R_map[l]*Pij[l])
                    changed = True

        # --- Detect violations ---
        for l in line_ids:
            overload = max(0.0, abs(Pij[l]) - Smax_map[l])
            sigma_fc_line[(l,t)] = overload
            sigma_fc_time[t] += overload

    return sigma_fc_line, sigma_fc_time



# --------------------------------------------------
# STEP 2 — Compute σ_fc using baseline fleet schedule
# --------------------------------------------------
sigma_fc_dict, sigma_fc = compute_sigma_fc_passive(Pfleet_baseline)


# --------------------------------------------------
# STEP 3 — Build congestion regime signal c_dict
# --------------------------------------------------
c_dict = {}
for t in T:
    total_fc = sigma_fc[t]
    if total_fc <= 1e-6:
        c_dict[t] = 0
    elif total_fc < THETA_REGIME[2]:
        c_dict[t] = 1
    else:
        c_dict[t] = 2


# ======================================================================
# SECTION 6 — TARIFF MODEL BUILD + SOLVE (NOW c_dict EXISTS)
# ======================================================================


m = build_model(
    c_init=c_dict,
    enforce_regime=True,
    sigma_fc_init=sigma_fc
)



res = solver.solve(m, tee=True)
term = res.solver.termination_condition
print("Tariff solve termination:", term)
if term not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible):
    m.write("tariff_infeasible.lp", io_options={"symbolic_solver_labels": True})
    raise RuntimeError(f"Tariff optimization infeasible: {term}")

# ======================================================================
# SECTION 7 — EXTRACT RESULTS + PLOTS (UNCHANGED, AFTER SOLVE)
# ======================================================================
T_list = list(m.T)

P_tariff      = np.array([value(m.P[t])        for t in T_list])
Pfleet_tariff = np.array([value(m.Pfleet[t])   for t in T_list])
lambda_tariff = np.array([value(m.lambda_t[t]) for t in T_list])


# -----------------------------
# Extract network results (tariff case)
# -----------------------------
if USE_NETWORK:
    V_tariff = np.array([
        [value(m.V[i, t]) for t in T_list]
        for i in m.I
    ])
else:
    V_tariff = None

# =========================================================
# PLOTS (CLEAN, NON-DUPLICATED, ECONOMICALLY MEANINGFUL)
# =========================================================


# Baseline net load (excluding fleet)
baseline_net_load = np.array([
    sum(P_load_dict[(i, t)] - PV_dict[(i, t)] for i in I)
    for t in T_list
])

total_load_baseline = baseline_net_load + Pfleet_baseline
total_load_tariff = baseline_net_load + Pfleet_tariff


# =========================================================
# HELPER
# =========================================================
def voltage_violation(V):
    return np.maximum(0, V_MIN - V) + np.maximum(0, V - V_MAX)


# =========================================================
# 1. Fleet charging + tariff
# =========================================================
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.step(T_list, P_tariff, where="post", lw=2, label="Fleet charging $P_t$")
ax1.set_xlabel("Hour")
ax1.set_ylabel("Fleet charging power (MW)")
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.step(T_list, lambda_tariff, where="post", lw=2, ls="--",
         label="Tariff $\\lambda_t$")
ax2.set_ylabel("Tariff (€/MWh)")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left")

plt.title("Fleet charging response to congestion tariff")
plt.tight_layout()
plt.show()


# =========================================================
# 2. Fleet charging: baseline vs tariff
# =========================================================
plt.figure(figsize=(12, 4))
plt.step(T_list, Pfleet_baseline, where="post", lw=2,
         color="black", label="Baseline (no tariff)")
plt.step(T_list, P_tariff, where="post", lw=2,
         color="#9467bd", label="Tariff (solution)")
plt.xlabel("Hour")
plt.ylabel("Fleet charging power (MW)")
plt.title("Fleet charging: baseline vs tariff")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =========================================================
# 3. Aggregate system load
# =========================================================
plt.figure(figsize=(12, 5))
plt.plot(T_list, total_load_baseline, "--", lw=3,
         color="black", label="Baseline (no tariff)")
plt.plot(T_list, total_load_tariff, lw=3,
         color="#1f77b4", label="Tariff (solution)")
plt.xlabel("Hour")
plt.ylabel("Total system load (MW)")
plt.title("Aggregate system load")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =========================================================
# Forecast vs realized overload
# =========================================================

# Forecast overload (passive PF, no flexibility)
sigma_fc_vec = np.array([sigma_fc[t] for t in T_list])

# Realized overload after tariff optimization
sigma_realized_vec = np.array([
    sum(value(m.sigma[l, t]) for l in m.L) for t in T_list
])

# Realized overload in baseline (fleet optimized, no tariff)
sigma_baseline_vec = np.array([
    sum(value(m_base.sigma[l, t]) for l in m_base.L) for t in T_list
])

plt.figure(figsize=(12,4))
plt.plot(T_list, sigma_fc_vec, lw=2, label="Forecast overload σ_fc (no response)")
plt.plot(T_list, sigma_baseline_vec, lw=2, ls=":", label="Baseline overload σ (fleet only)")
plt.plot(T_list, sigma_realized_vec, lw=2, ls="--", label="Tariff overload σ (after response)")
plt.legend()
plt.grid(alpha=0.3)
plt.title("Forecast vs realized thermal overload")
plt.tight_layout()
plt.show()

print("\n=== Forecast Miss Detection ===")
for t in T_list:
    realized = sum(value(m.sigma[l,t]) for l in m.L)
    forecast = sigma_fc[t]
    if realized > 1e-3 and forecast < 1e-6:
        print(f"Unexpected congestion at hour {t}: realized σ={realized:.3f}, forecast σ_fc={forecast:.3f}")

# =========================================================
# 5. Voltage violations
# =========================================================
if USE_NETWORK and V_tariff is not None:
    vviol = voltage_violation(V_tariff)

    plt.figure(figsize=(12, 4))
    plt.plot(T_list, vviol.sum(axis=0), lw=2)
    plt.xlabel("Hour")
    plt.ylabel("Total voltage violation")
    plt.title("Voltage violations (tariff solution)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    #HEATMAP

    plt.figure(figsize=(10, 6))
    plt.imshow(vviol, aspect="auto", cmap="Reds", origin="lower")
    plt.colorbar(label="Voltage violation")
    plt.xlabel("Hour")
    plt.ylabel("Bus index")
    plt.title("Voltage violation heat map")
    plt.tight_layout()
    plt.show()


# =========================================================
# WELFARE COMPARISON
# =========================================================
fleet_cost_baseline = sum(
    price_dict[t] * Pfleet_baseline[k] * DELTA_T
    for k, t in enumerate(T_list)
)


fleet_cost_tariff = sum(
    (price_dict[t] + value(m.lambda_t[t])) * value(m.P[t]) * DELTA_T
    for t in m.T
)

ENS_cost_tariff = (
    sum(C_ENS * value(m.ENS[i, t]) for i in m.I for t in m.T)
    if USE_NETWORK else 0.0
)

volt_cost_tariff = (
    sum(C_VOLT * (value(m.xi_low[i, t]) + value(m.xi_high[i, t]))
        for i in m.I for t in m.T)
    if USE_NETWORK else 0.0
)

welfare_df = pd.DataFrame({
    "Baseline (no tariff)": {
        "Fleet energy cost (€)": fleet_cost_baseline,
    },
    "Tariff (solution)": {
        "Fleet energy cost (€)": fleet_cost_tariff,
        "ENS cost (€)": ENS_cost_tariff,
        "Voltage slack cost (€)": volt_cost_tariff,
        "Total system cost (€)": fleet_cost_tariff + ENS_cost_tariff + volt_cost_tariff,
    }
})

print("\n=== Welfare summary ===")
print(welfare_df.round(2))


# =========================================================
# SOC FLEXIBILITY ENVELOPE
# =========================================================
T_hours = np.array(T_list)
SOC_init = E0_fleet / Emax_fleet

SOC_early = np.full_like(T_hours, np.nan, dtype=float)
soc = SOC_init
for k, t in enumerate(T_hours):
    soc = min(1.0, soc + eta_fleet * Pmax_fleet * DELTA_T / Emax_fleet)
    SOC_early[k] = soc

SOC_late = np.full_like(T_hours, np.nan, dtype=float)
soc = SOC_MIN_DEP
for k in reversed(range(len(T_hours))):
    SOC_late[k] = soc
    soc = max(SOC_init,
              soc - eta_fleet * Pmax_fleet * DELTA_T / Emax_fleet)

SOC_baseline = np.cumsum(
    eta_fleet * Pfleet_baseline * DELTA_T / Emax_fleet
) + SOC_init

SOC_opt = np.array([value(m.E[t]) / Emax_fleet for t in T_list])

plt.figure(figsize=(10, 6))
plt.fill_between(T_hours, SOC_late, SOC_early, alpha=0.4,
                 label="Flexibility envelope")
plt.plot(T_hours, SOC_baseline, lw=2, label="Baseline SOC")
plt.plot(T_hours, SOC_opt, lw=2.5, label="Optimized SOC")
plt.axhline(SOC_MIN_DEP, ls=":", lw=1.5, color="black")
plt.xlabel("Hour")
plt.ylabel("State of Charge")
plt.title("Fleet SOC flexibility envelope")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



#FLEET COST VS NETWORK STRESS
# --- Network stress metrics ---
sigma_sum_base = np.array([sum(pyo.value(m_base.sigma[l,t]) for l in m_base.L) for t in T_list])
sigma_sum_tar  = np.array([sum(pyo.value(m.sigma[l,t])      for l in m.L)      for t in T_list])

V_base = np.array([[pyo.value(m_base.V[i,t]) for t in T_list] for i in m_base.I])
V_tar  = np.array([[pyo.value(m.V[i,t])      for t in T_list] for i in m.I])

def vviol_total(V):
    return np.maximum(0, V_MIN - V) + np.maximum(0, V - V_MAX)

vviol_base_sum = vviol_total(V_base).sum()
vviol_tar_sum  = vviol_total(V_tar).sum()

stress_base = sigma_sum_base.sum() + vviol_base_sum
stress_tar  = sigma_sum_tar.sum()  + vviol_tar_sum

# --- Fleet private cost ---
fleet_cost_base = sum(price_dict[t] * pyo.value(m_base.P[t]) * DELTA_T for t in T_list)
fleet_cost_tar  = sum((price_dict[t] + pyo.value(m.lambda_t[t])) * pyo.value(m.P[t]) * DELTA_T for t in T_list)

plt.figure(figsize=(7,5))
plt.scatter([fleet_cost_base, fleet_cost_tar], [stress_base, stress_tar], s=120)
plt.text(fleet_cost_base, stress_base, "Baseline", va="bottom")
plt.text(fleet_cost_tar,  stress_tar,  "Tariff",   va="bottom")
plt.xlabel("Fleet private cost (€)")
plt.ylabel("Network stress (Σ overload slack + Σ voltage violation)")
plt.title("Trade-off: fleet cost vs network stress")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()





#THERMAL CONSTRAINT VIOLATIONS
sigma_base = np.array([sum(pyo.value(m_base.sigma[l,t]) for l in m_base.L) for t in T_list])
sigma_tar  = np.array([sum(pyo.value(m.sigma[l,t])      for l in m.L)      for t in T_list])

plt.figure(figsize=(12,4))
plt.plot(T_list, sigma_base, "--", lw=3, label="Baseline thermal slack Σσ (no tariff)")
plt.plot(T_list, sigma_tar,  lw=3, label="Tariff thermal slack Σσ (solution)")
plt.xlabel("Hour")
plt.ylabel("Overload slack (MW)")
plt.title("Thermal constraint violations: baseline vs tariff")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

V_base = np.array([[pyo.value(m_base.V[i,t]) for t in T_list] for i in m_base.I])
V_tar  = np.array([[pyo.value(m.V[i,t])      for t in T_list] for i in m.I])

vviol_base = vviol_total(V_base)
vviol_tar  = vviol_total(V_tar)

plt.figure(figsize=(12,4))
plt.plot(T_list, vviol_base.sum(axis=0), "--", lw=3, label="Baseline voltage violation")
plt.plot(T_list, vviol_tar.sum(axis=0),  lw=3, label="Tariff voltage violation")
plt.xlabel("Hour")
plt.ylabel("Total voltage violation")
plt.title("Voltage violations: baseline vs tariff")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

vmax = max(vviol_base.max(), vviol_tar.max())

plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.imshow(vviol_base, aspect="auto", origin="lower", cmap="Reds", vmin=0, vmax=vmax)
plt.title("Baseline voltage violation heatmap")
plt.xlabel("Hour"); plt.ylabel("Bus index")
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(vviol_tar, aspect="auto", origin="lower", cmap="Reds", vmin=0, vmax=vmax)
plt.title("Tariff voltage violation heatmap")
plt.xlabel("Hour"); plt.ylabel("Bus index")
plt.colorbar()

plt.tight_layout()
plt.show()


#OKAY
thermal_base = sigma_base.sum()
thermal_tar  = sigma_tar.sum()

volt_base = vviol_base.sum()
volt_tar  = vviol_tar.sum()

red_thermal = 100*(thermal_base - thermal_tar)/max(1e-9, thermal_base)
red_volt    = 100*(volt_base    - volt_tar   )/max(1e-9, volt_base)

plt.figure(figsize=(7,5))
plt.bar(["Thermal (Σσ)", "Voltage (Σviol)"], [red_thermal, red_volt])
plt.ylabel("Reduction (%)")
plt.title("Extent of violation reduction due to tariff")
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

print(f"Thermal slack reduction: {red_thermal:.1f}%")
print(f"Voltage violation reduction: {red_volt:.1f}%")




import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import value

# ===========================
# TIME AXIS
# ===========================
T_list = list(m.T)               # daily: 0..23 or session indices
x = np.array(T_list, dtype=int)

# If session-based:
# labels = [f"{T_clock[k]}:00" for k in T_list]
# If daily-based:
labels = [str(t) for t in T_list]

# ===========================
# SERIES TO PLOT
# ===========================
# 1) Forecast congestion σ_fc (exogenous signal)
# if you have sigma_fc as dict: sigma_fc[t]
sigma_fc_vec = np.array([float(sigma_fc[t]) for t in T_list])

# OR if you have sigma_fc_sum array already aligned to T_list:
# sigma_fc_vec = np.array(sigma_fc_sum, dtype=float)

# 2) Tariff λ from optimization output
lam_vec = np.array([value(m.lambda_t[t]) for t in T_list], dtype=float)

# 3) Realized overload slack σ from solved network constraints
sigma_realized_vec = np.array([
    sum(value(m.sigma[l, t]) for l in m.L) for t in T_list
], dtype=float)

# Optional: baseline realized overload slack
sigma_base_vec = np.array([
    sum(value(m_base.sigma[l, t]) for l in m_base.L) for t in T_list
], dtype=float)

# ===========================
# REGIME SHADING (based on σ_fc)
# ===========================
EPS_CONG = 1e-6
THETA2 = THETA_REGIME[2]

regime = np.zeros(len(T_list), dtype=int)
for i, t in enumerate(T_list):
    s = sigma_fc_vec[i]
    if s <= EPS_CONG:
        regime[i] = 0   # normal
    elif s < THETA2:
        regime[i] = 1   # critical
    else:
        regime[i] = 2   # congested

# ===========================
# PLOT
# ===========================
fig, ax1 = plt.subplots(figsize=(13, 5))

# --- background regime shading ---
for i in range(len(T_list)):
    if regime[i] == 1:
        ax1.axvspan(x[i]-0.5, x[i]+0.5, alpha=0.10)  # critical
    elif regime[i] == 2:
        ax1.axvspan(x[i]-0.5, x[i]+0.5, alpha=0.18)  # congested

# --- left axis: congestion indicators ---
ax1.plot(x, sigma_fc_vec, lw=2.5, label=r"Forecast overload $\sigma^{fc}_t$")
ax1.plot(x, sigma_realized_vec, lw=2.5, ls="--", label=r"Realized overload $\sigma_t$ (tariff)")
ax1.plot(x, sigma_base_vec, lw=2.5, ls=":", label=r"Realized overload $\sigma_t$ (baseline)")

ax1.set_xlabel("Time step")
ax1.set_ylabel(r"Thermal overload slack (MW)")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45)
ax1.grid(alpha=0.3)

# --- right axis: tariff ---
ax2 = ax1.twinx()
ax2.step(x, lam_vec, where="post", lw=2.5, label=r"Tariff $\lambda_t$")
ax2.set_ylabel(r"Tariff (€/MWh)")

# --- legends ---
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left")

plt.title("Mechanism check: forecast congestion → tariff signal → realized congestion")
plt.tight_layout()
plt.show()

P_base = np.array([value(m_base.Pfleet[t]) for t in T_list], dtype=float)
P_tar  = np.array([value(m.Pfleet[t]) for t in T_list], dtype=float)

plt.figure(figsize=(13, 4))
plt.step(x, P_base, where="post", lw=2.5, label="Baseline charging")
plt.step(x, P_tar,  where="post", lw=2.5, label="Tariff charging")

# Shade hours forecast-congested (based on σ_fc)
for i in range(len(T_list)):
    if regime[i] == 2:
        plt.axvspan(x[i]-0.5, x[i]+0.5, alpha=0.15)


mismatch = sigma_realized_vec - sigma_fc_vec

plt.figure(figsize=(13,4))
plt.bar(x, mismatch)
plt.axhline(0, lw=2)
plt.xticks(x, labels, rotation=45)
plt.xlabel("Time step")
plt.ylabel(r"$\sigma_t - \sigma^{fc}_t$ (MW)")
plt.title("Forecast vs realized overload mismatch (after fleet response)")
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.show()


print("Baseline s_dep:", pyo.value(m_base.s_dep))
print("Tariff s_dep:",    pyo.value(m.s_dep))
print("Baseline E_last:", pyo.value(m_base.E[t_last_fleet]), " target:", Etarget_fleet)
print("Tariff E_last:",   pyo.value(m.E[t_last_fleet]),      " target:", Etarget_fleet)

print("Max baseline charging:", max(pyo.value(m_base.P[t]) for t in m_base.T))
print("End SOC baseline:", pyo.value(m_base.E[t_last_fleet]) / Emax_fleet)
print("Any nonzero prices?", any(price_dict[t] != 0 for t in T))

print("Total PV curtailment:", sum(pyo.value(m.PVcurt[i,t]) for i in m.I for t in m.T))
print("Total ENS:",           sum(pyo.value(m.ENS[i,t])     for i in m.I for t in m.T))

print("Tariff s_dep:", pyo.value(m.s_dep))
print("Tariff end SOC:", pyo.value(m.E[t_last_fleet])/Emax_fleet)


volt_slack_sum = sum(pyo.value(m.xi_low[i,t] + m.xi_high[i,t]) for i in m.I for t in m.T)
print("Total voltage slack:", volt_slack_sum)

print([(t, a_fleet[t]) for t in T])

ens_total = sum(value(m.ENS[i,t]) for i in m.I for t in m.T)
curt_total = sum(value(m.PVcurt[i,t]) for i in m.I for t in m.T)
print("ENS total:", ens_total)
print("PVcurt total:", curt_total)

deltaP = sum(abs(value(m.Pfleet[t]) - value(m_base.Pfleet[t])) for t in m.T)
print("Fleet adjustment L1:", deltaP)


import copy
import numpy as np
import pyomo.environ as pyo
from pyomo.environ import value

def total_sigma(model):
    return float(sum(value(model.sigma[l,t]) for l in model.L for t in model.T))

def solve_tariff_variant(fix_fleet=False, fix_ens=False, fix_pvcurt=False):
    # build fresh model (important: new instance)
    mm = build_model(c_init=c_dict, enforce_regime=True, sigma_fc_init=sigma_fc)

    # Optionally FIX fleet to baseline schedule (removes fleet as an actuator)
    if fix_fleet:
        for k, t in enumerate(list(mm.T)):
            mm.Pfleet[t].fix(float(Pfleet_baseline[k]))
            mm.P[t].fix(float(Pfleet_baseline[k]))  # because PFLEET_LINK enforces equality anyway

    # Optionally FIX ENS and/or PV curtailment
    if fix_ens:
        for i in mm.I:
            for t in mm.T:
                mm.ENS[i,t].fix(0.0)

    if fix_pvcurt:
        for i in mm.I:
            for t in mm.T:
                mm.PVcurt[i,t].fix(0.0)

    res = solver.solve(mm, tee=False)
    tc = res.solver.termination_condition
    if tc not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible):
        return None, tc

    return mm, tc

# --- Baseline overload ---
sigma_base_total = float(sum(value(m_base.sigma[l,t]) for l in m_base.L for t in m_base.T))

# --- Full tariff (all actuators) ---
m_full, tc_full = solve_tariff_variant(fix_fleet=False, fix_ens=False, fix_pvcurt=False)
sigma_full_total = total_sigma(m_full)

# --- Remove each actuator one at a time ---
m_noFleet, tc_nf = solve_tariff_variant(fix_fleet=True,  fix_ens=False, fix_pvcurt=False)
m_noENS,   tc_ne = solve_tariff_variant(fix_fleet=False, fix_ens=True,  fix_pvcurt=False)
m_noPV,    tc_np = solve_tariff_variant(fix_fleet=False, fix_ens=False, fix_pvcurt=True)

# --- Fleet-only (no ENS, no PVcurt) ---
m_fleetonly, tc_fo = solve_tariff_variant(fix_fleet=False, fix_ens=True, fix_pvcurt=True)

# compute totals (guard infeasibility)
def safe_sigma(model):
    return None if model is None else total_sigma(model)

sigma_noFleet = safe_sigma(m_noFleet)
sigma_noENS   = safe_sigma(m_noENS)
sigma_noPV    = safe_sigma(m_noPV)
sigma_fleetonly = safe_sigma(m_fleetonly)

print("\n=== Thermal overload attribution (counterfactuals) ===")
print("Σσ baseline:", sigma_base_total)
print("Σσ full tariff:", sigma_full_total)

print("\nVariants (Σσ):")
print("No Fleet (fleet fixed to baseline):", sigma_noFleet, " status:", tc_nf)
print("No ENS (ENS fixed to 0):", sigma_noENS, " status:", tc_ne)
print("No PVcurt (PVcurt fixed to 0):", sigma_noPV, " status:", tc_np)
print("Fleet-only (ENS=0, PVcurt=0):", sigma_fleetonly, " status:", tc_fo)

# marginal contributions vs full tariff (only if feasible)
def marginal(sigma_variant):
    return None if sigma_variant is None else float(sigma_variant - sigma_full_total)

print("\nMarginal contribution to relieving overload (ΔΣσ when removed):")
print("Fleet:", marginal(sigma_noFleet))
print("ENS:", marginal(sigma_noENS))
print("PVcurt:", marginal(sigma_noPV))



import numpy as np
import matplotlib.pyplot as plt

lines = list(m.L)
times = list(m.T)

sigma_matrix = np.array([
    [value(m.sigma[l, t]) for t in times]
    for l in lines
])

plt.figure(figsize=(12,6))
plt.imshow(sigma_matrix, aspect="auto", origin="lower", cmap="Reds")
plt.colorbar(label="Thermal overload slack (MW)")
plt.yticks(range(len(lines)), lines)
plt.xticks(range(len(times)), times)
plt.xlabel("Hour")
plt.ylabel("Line ID")
plt.title("Line thermal congestion over time")
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(10,5))

bars = ax.bar(lines, [0]*len(lines))
ax.set_ylim(0, max(value(m.sigma[l,t]) for l in lines for t in times)*1.1)
ax.set_xlabel("Line ID")
ax.set_ylabel("Thermal overload slack (MW)")

def update(frame):
    t = times[frame]
    vals = [value(m.sigma[l,t]) for l in lines]
    for bar, v in zip(bars, vals):
        bar.set_height(v)
    ax.set_title(f"Thermal congestion at hour {t}")
    return bars

ani = animation.FuncAnimation(fig, update, frames=len(times), repeat=True)

plt.show()
