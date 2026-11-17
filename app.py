"""
HEAT TRANSFER STUDIO - Sistema Integrado de Engenharia Térmica (BETA)
Desenvolvido para Fenômenos de Transporte II / Transferência de Calor (EEL-USP)
Frameworks: Streamlit, Plotly, Pandas, NumPy, SciPy
"""

import math
import re
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configuração da Página Streamlit
st.set_page_config(
    page_title="Heat Transfer Studio - EEL/USP",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. BANCO DE PROPRIEDADES TERMOFÍSICAS E MATERIAIS
# ==========================================

SOLID_MATERIALS = {
    "Personalizado": {"k": 50.0, "rho": 7800.0, "cp": 500.0},
    "Cobre Puro (300 K)": {"k": 401.0, "rho": 8933.0, "cp": 385.0},
    "Alumínio 6061 (300 K)": {"k": 167.0, "rho": 2700.0, "cp": 896.0},
    "Aço Carbono AISI 1010": {"k": 63.9, "rho": 7832.0, "cp": 434.0},
    "Aço Inoxidável AISI 304": {"k": 14.9, "rho": 7900.0, "cp": 477.0},
    "Tijolo Refratário": {"k": 1.10, "rho": 2000.0, "cp": 960.0},
    "Concreto Comum": {"k": 1.40, "rho": 2300.0, "cp": 880.0},
    "Lã de Vidro (Isolante)": {"k": 0.038, "rho": 32.0, "cp": 835.0},
    "Poliestireno Expandido (Isopor)": {"k": 0.027, "rho": 25.0, "cp": 1210.0},
    "Gelo (273 K)": {"k": 1.88, "rho": 920.0, "cp": 2040.0}
}


class WaterProperties:
    """Correlações para propriedades da água líquida (0 °C a 300 °C)."""

    @staticmethod
    def density(T_C: float) -> float:
        T = np.clip(T_C, 0.0, 350.0)
        rho = 1000.0 / (1.0 + 1.8e-4 * (T - 4.0)**2) if T < 100 else 1000.0 * (1.0 - 0.002 * (T - 100)**0.8)
        return float(max(rho, 500.0))

    @staticmethod
    def cp(T_C: float) -> float:
        T = np.clip(T_C, 0.0, 300.0)
        return float(4184.0 + 0.4 * T + 0.002 * T**2)

    @staticmethod
    def k(T_C: float) -> float:
        T = np.clip(T_C, 0.0, 300.0)
        return float(0.569 + 0.0019 * T - 7.0e-6 * T**2)

    @staticmethod
    def mu(T_C: float) -> float:
        T = max(T_C, 0.01)
        return float(2.414e-5 * 10**(247.8 / (T + 30.0)))

    @staticmethod
    def pr(T_C: float) -> float:
        return float((WaterProperties.cp(T_C) * WaterProperties.mu(T_C)) / WaterProperties.k(T_C))


class AirProperties:
    """Correlações para propriedades do Ar a 1 atm."""

    @staticmethod
    def density(T_C: float) -> float:
        T_K = T_C + 273.15
        return float(101325.0 / (287.05 * T_K))

    @staticmethod
    def cp(T_C: float) -> float:
        return 1005.0

    @staticmethod
    def k(T_C: float) -> float:
        T_K = T_C + 273.15
        return float(0.02424 * (T_K / 293.15)**0.8)

    @staticmethod
    def mu(T_C: float) -> float:
        T_K = T_C + 273.15
        return float(1.81e-5 * (T_K / 293.15)**0.7)

    @staticmethod
    def pr(T_C: float) -> float:
        return 0.707


# ==========================================
# 2. MÓDULOS DE CÁLCULO DE ENGENHARIA
# ==========================================

class ConductionSolver:
    """Solver para condução de calor em regime permanente (Plana, Cilíndrica e Esférica)."""

    @staticmethod
    def plane_wall(k: float, A: float, L: float, T1: float, T2: float) -> Dict:
        R_t = L / (k * A)
        q = (T1 - T2) / R_t
        flux = q / A
        return {"R_t": R_t, "q": q, "flux": flux, "geom": "Plana"}

    @staticmethod
    def cylindrical_wall(k: float, L_cyl: float, r_i: float, r_o: float, Ti: float, To: float) -> Dict:
        R_t = math.log(r_o / r_i) / (2 * math.pi * k * L_cyl)
        q = (Ti - To) / R_t
        return {"R_t": R_t, "q": q, "geom": "Cilíndrica"}

    @staticmethod
    def spherical_wall(k: float, r_i: float, r_o: float, Ti: float, To: float) -> Dict:
        R_t = (1.0 / r_i - 1.0 / r_o) / (4 * math.pi * k)
        q = (Ti - To) / R_t
        return {"R_t": R_t, "q": q, "geom": "Esférica"}


class ConvectionSolver:
    """Solver para convecção forçada e natural."""

    @staticmethod
    def internal_pipe(D: float, L: float, V: float, T_m: float, fluid_type: str = "Água", heating: bool = True) -> Dict:
        if fluid_type == "Água":
            rho, mu, k, pr = WaterProperties.density(T_m), WaterProperties.mu(T_m), WaterProperties.k(T_m), WaterProperties.pr(T_m)
        else:
            rho, mu, k, pr = AirProperties.density(T_m), AirProperties.mu(T_m), AirProperties.k(T_m), AirProperties.pr(T_m)

        Re = (rho * V * D) / mu
        if Re < 2300:
            regime = "Laminar"
            Nu = 3.66
        elif Re < 10000:
            regime = "Transição"
            # Correlação de Gnielinski
            f = (0.790 * math.log(Re) - 1.64)**(-2)
            Nu = ((f / 8.0) * (Re - 1000.0) * pr) / (1.0 + 12.7 * ((f / 8.0)**0.5) * (pr**(2.0/3.0) - 1.0))
        else:
            regime = "Turbulento (Dittus-Boelter)"
            n = 0.4 if heating else 0.3
            Nu = 0.023 * (Re**0.8) * (pr**n)

        h = (Nu * k) / D
        return {"Re": Re, "Nu": Nu, "h": h, "regime": regime, "Pr": pr}

    @staticmethod
    def external_flat_plate(L: float, V: float, T_surface: float, T_inf: float, fluid_type: str = "Ar") -> Dict:
        T_film = (T_surface + T_inf) / 2.0
        if fluid_type == "Ar":
            rho, mu, k, pr = AirProperties.density(T_film), AirProperties.mu(T_film), AirProperties.k(T_film), AirProperties.pr(T_film)
        else:
            rho, mu, k, pr = WaterProperties.density(T_film), WaterProperties.mu(T_film), WaterProperties.k(T_film), WaterProperties.pr(T_film)

        nu_kin = mu / rho
        Re = (V * L) / nu_kin
        regime = "Laminar" if Re < 5e5 else ("Turbulento" if Re > 1e6 else "Misto (Laminar + Turbulento)")

        if Re < 5e5:
            Nu = 0.664 * (Re**0.5) * (pr**(1.0/3.0))
        else:
            Nu = (0.037 * (Re**0.8) - 871.0) * (pr**(1.0/3.0))

        h = (Nu * k) / L
        q_flux = h * (T_surface - T_inf)
        return {"Re": Re, "Nu": Nu, "h": h, "flux": q_flux, "regime": regime, "T_film": T_film}

    @staticmethod
    def natural_convection_vertical_plate(L: float, T_surface: float, T_inf: float, fluid_type: str = "Ar") -> Dict:
        T_film = (T_surface + T_inf) / 2.0
        beta = 1.0 / (T_film + 273.15)
        g = 9.81
        delta_T = abs(T_surface - T_inf)

        if fluid_type == "Ar":
            rho, mu, k, pr = AirProperties.density(T_film), AirProperties.mu(T_film), AirProperties.k(T_film), AirProperties.pr(T_film)
        else:
            rho, mu, k, pr = WaterProperties.density(T_film), WaterProperties.mu(T_film), WaterProperties.k(T_film), WaterProperties.pr(T_film)

        nu = mu / rho
        Gr = (g * beta * delta_T * (L**3)) / (nu**2)
        Ra = Gr * pr

        if Ra < 1e9:
            Nu = 0.68 + (0.670 * (Ra**0.25)) / ((1.0 + (0.492 / pr)**(9.0/16.0))**(4.0/9.0))
            regime = "Laminar"
        else:
            Nu = (0.825 + (0.387 * (Ra**(1.0/6.0))) / ((1.0 + (0.492 / pr)**(9.0/16.0))**(8.0/27.0)))**2
            regime = "Turbulento"

        h = (Nu * k) / L
        q_flux = h * delta_T
        return {"Gr": Gr, "Ra": Ra, "Nu": Nu, "h": h, "flux": q_flux, "regime": regime}


class FinSolver:
    """Solver completo para superfícies estendidas (Aletas retangulares de seção uniforme - Incropera Tabela 3.4)."""

    @staticmethod
    def rectangular_fin(k: float, t: float, w: float, L: float, h: float, Tb: float, Tinf: float, bc_type: str = "Extremidade Convectiva", TL: float = 40.0) -> Dict:
        P = 2.0 * (w + t)
        Ac = w * t
        m = math.sqrt((h * P) / (k * Ac))
        M = math.sqrt(h * P * k * Ac) * (Tb - Tinf)
        ml = m * L
        theta_b = Tb - Tinf
        theta_L = TL - Tinf

        x_vals = np.linspace(0, L, 100)

        if bc_type == "Extremidade Convectiva":
            ratio = h / (m * k)
            num = math.sinh(ml) + ratio * math.cosh(ml)
            den = math.cosh(ml) + ratio * math.sinh(ml)
            q_fin = M * (num / den)
            T_vals = Tinf + theta_b * (np.cosh(m * (L - x_vals)) + ratio * np.sinh(m * (L - x_vals))) / den
            A_fin = P * L + Ac

        elif bc_type == "Extremidade Adiabática":
            q_fin = M * math.tanh(ml)
            T_vals = Tinf + theta_b * (np.cosh(m * (L - x_vals)) / math.cosh(ml))
            A_fin = P * L

        elif bc_type == "Temperatura Prescrita na Ponta":
            q_fin = M * ((math.cosh(ml) - (theta_L / theta_b)) / math.sinh(ml))
            T_vals = Tinf + theta_b * (((theta_L / theta_b) * np.sinh(m * x_vals) + np.sinh(m * (L - x_vals))) / math.sinh(ml))
            A_fin = P * L

        else:  # Aleta Infinita
            q_fin = M
            T_vals = Tinf + theta_b * np.exp(-m * x_vals)
            A_fin = P * L

        q_max = h * A_fin * theta_b
        eta = q_fin / q_max if q_max > 0 else 0.0

        # Efetividade (epsilon_f = q_fin / (h * Ac * theta_b))
        q_no_fin = h * Ac * theta_b
        effectiveness = q_fin / q_no_fin if q_no_fin > 0 else 0.0

        return {
            "m": m,
            "q_fin": q_fin,
            "eta": eta,
            "effectiveness": effectiveness,
            "x_vals": x_vals,
            "T_vals": T_vals
        }


class TransientSolver:
    """Solver para regime transiente (Método da Capacitância Global e Biot)."""

    @staticmethod
    def lumped_capacitance(rho: float, V: float, cp: float, As: float, h: float, k: float, Ti: float, Tinf: float, t_final: float) -> Tuple[pd.DataFrame, float, float]:
        Lc = V / As
        biot = (h * Lc) / k
        tau = (rho * V * cp) / (h * As)  # Constante de tempo térmica (s)

        t_vals = np.linspace(0, t_final, 150)
        T_vals = Tinf + (Ti - Tinf) * np.exp(-t_vals / tau)

        df = pd.DataFrame({"Tempo (s)": t_vals, "Temperatura (°C)": T_vals})
        return df, biot, tau


class HeatExchangerSolver:
    """Solver para Trocadores de Calor (Método MLDT e Método e-NTU)."""

    @staticmethod
    def lmtd_analysis(Thi: float, Tho: float, Tci: float, Tco: float, flow_type: str = "Contracorrente") -> Dict:
        dt1 = Thi - Tco if flow_type == "Contracorrente" else Thi - Tci
        dt2 = Tho - Tci if flow_type == "Contracorrente" else Tho - Tco

        if abs(dt1 - dt2) < 1e-4:
            lmtd = dt1
        elif dt1 > 0 and dt2 > 0:
            lmtd = (dt1 - dt2) / math.log(dt1 / dt2)
        else:
            lmtd = 0.0

        return {"LMTD": lmtd, "DT1": dt1, "DT2": dt2}

    @staticmethod
    def ntu_analysis(Ch: float, Cc: float, UA: float, Thi: float, Tci: float, config: str = "Contracorrente") -> Dict:
        Cmin = min(Ch, Cc)
        Cmax = max(Ch, Cc)
        Cr = Cmin / Cmax if Cmax > 0 else 0.0
        NTU = UA / Cmin if Cmin > 0 else 0.0

        if Cr == 0:
            eps = 1.0 - math.exp(-NTU)
        elif config == "Contracorrente":
            if abs(Cr - 1.0) < 1e-4:
                eps = NTU / (1.0 + NTU)
            else:
                eps = (1.0 - math.exp(-NTU * (1.0 - Cr))) / (1.0 - Cr * math.exp(-NTU * (1.0 - Cr)))
        elif config == "Correntes Paralelas":
            eps = (1.0 - math.exp(-NTU * (1.0 + Cr))) / (1.0 + Cr)
        else:  # Casco e Tubos (1 passe casco)
            term = math.sqrt(1.0 + Cr**2)
            eps = 2.0 * (1.0 + Cr + term * ((1.0 + math.exp(-NTU * term)) / (1.0 - math.exp(-NTU * term))))**(-1)

        q_max = Cmin * (Thi - Tci)
        q = eps * q_max
        Tho = Thi - (q / Ch)
        Tco = Tci + (q / Cc)

        return {"Cmin": Cmin, "Cmax": Cmax, "Cr": Cr, "NTU": NTU, "eps": eps, "q": q, "q_max": q_max, "Tho": Tho, "Tco": Tco}


class RadiationSolver:
    """Solver para Radiação Térmica (Lei de Stefan-Boltzmann e Troca Radiativa)."""
    SIGMA = 5.670374419e-8  # W/(m²·K⁴)

    @staticmethod
    def surface_surroundings(eps: float, A: float, Ts_C: float, Tsur_C: float) -> Dict:
        Ts_K = Ts_C + 273.15
        Tsur_K = Tsur_C + 273.15

        Eb = RadiationSolver.SIGMA * (Ts_K**4)
        E = eps * Eb
        q_rad = eps * RadiationSolver.SIGMA * A * (Ts_K**4 - Tsur_K**4)
        h_rad = eps * RadiationSolver.SIGMA * (Ts_K + Tsur_K) * (Ts_K**2 + Tsur_K**2)

        return {"Eb": Eb, "E": E, "q_rad": q_rad, "h_rad": h_rad}


# ==========================================
# 3. SOLUCIONADOR AUTOMÁTICO DE ENUNCIADOS
# ==========================================

class AutoSolver:
    @staticmethod
    def parse_and_solve(text: str) -> str:
        text_lower = text.lower()
        results = ["### 🔍 Análise Automática do Enunciado", f"**Texto Analisado:** *\"{text}\"*"]

        temps = [float(t) for t in re.findall(r'(\d+(?:\.\d+)?)\s*(?:°c|c|graus)', text_lower)]

        if "aleta" in text_lower:
            results.append("**Tipo de Problema Detectado:** Superfícies Estendidas (Aletas).")
            results.append("- Utilize a aba **Aletas** para configurar as dimensões ($L, w, t$) e o tipo de ponta (convectiva, adiabática, prescrita ou infinita).")
        elif "trocador" in text_lower or "lmtd" in text_lower or "ntu" in text_lower:
            results.append("**Tipo de Problema Detectado:** Trocador de Calor.")
            if len(temps) >= 4:
                res = HeatExchangerSolver.lmtd_analysis(temps[0], temps[1], temps[2], temps[3])
                results.append(f"- Temperaturas detectadas: {temps[:4]} °C")
                results.append(f"- **MLDT (Contracorrente):** `{res['LMTD']:.2f} °C`")
        elif "parede" in text_lower or "placa" in text_lower:
            results.append("**Tipo de Problema Detectado:** Condução em Parede Plana ou Convecção Externa.")
            if len(temps) >= 2:
                results.append(f"- Temperaturas detectadas: $T_1 = {temps[0]}°C$, $T_2 = {temps[1]}°C$.")
        elif "tubo" in text_lower or "cilindro" in text_lower:
            results.append("**Tipo de Problema Detectado:** Escoamento Interno ou Condução Cilíndrica.")
        elif "radiação" in text_lower or "emissividade" in text_lower:
            results.append("**Tipo de Problema Detectado:** Radiação Térmica (Stefan-Boltzmann).")
        else:
            results.append("⚠️ Selecione o módulo específico no menu lateral para inserir os parâmetros numéricos completos.")

        return "\n".join(results)


# ==========================================
# 4. INTERFACE GRÁFICA (STREAMLIT)
# ==========================================

def main():
    st.sidebar.title("🔥 Heat Transfer Studio")
    st.sidebar.caption("Versão BETA • Engenharia Térmica")
    st.sidebar.markdown("---")

    menu = st.sidebar.selectbox(
        "Navegação Principal",
        [
            "Início",
            "Banco de Propriedades",
            "Condução",
            "Resistências Térmicas",
            "Convecção Forçada Interna",
            "Convecção Forçada Externa",
            "Convecção Natural",
            "Aletas",
            "Regime Transiente",
            "Trocadores de Calor",
            "Radiação Térmica",
            "Resolver Exercício",
            "Sobre"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.info("Escola de Engenharia de Lorena - USP\nFenômenos de Transporte II")

    if menu == "Início":
        st.title("🔥 Heat Transfer Studio")
        st.markdown("### Plataforma Integrada de Engenharia Térmica e Fenômenos de Transporte")
        st.markdown("Ferramenta computacional desenvolvida para simulação, dimensionamento e resolução de problemas de transferência de calor.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Condução & Aletas", "4 Cond. Contorno", "Incropera")
        c2.metric("Convecção", "Interna/Externa/Nat.", "Auto-Regime")
        c3.metric("Trocadores", "MLDT & ε-NTU", "Completo")
        c4.metric("Radiação & Trans.", "Biot & Stefan-Boltz.", "Ativo")

    elif menu == "Banco de Propriedades":
        st.title("📊 Banco de Propriedades Termofísicas")
        tab1, tab2 = st.tabs(["Fluidos (Água e Ar)", "Materiais Sólidos"])

        with tab1:
            fluid = st.selectbox("Selecione o Fluido", ["Água", "Ar"])
            T = st.slider("Temperatura (°C)", 0.0, 300.0, 25.0)

            if fluid == "Água":
                rho, cp, k, mu, pr = WaterProperties.density(T), WaterProperties.cp(T), WaterProperties.k(T), WaterProperties.mu(T), WaterProperties.pr(T)
            else:
                rho, cp, k, mu, pr = AirProperties.density(T), AirProperties.cp(T), AirProperties.k(T), AirProperties.mu(T), AirProperties.pr(T)

            col1, col2, col3 = st.columns(3)
            col1.metric("Massa Específica ($\\rho$)", f"{rho:.3f} kg/m³")
            col2.metric("Condutividade Térmica ($k$)", f"{k:.4f} W/(m·K)")
            col3.metric("Número de Prandtl ($Pr$)", f"{pr:.3f}")

            col4, col5 = st.columns(2)
            col4.metric("Calor Específico ($c_p$)", f"{cp:.1f} J/(kg·K)")
            col5.metric("Viscosidade Dinâmica ($\\mu$)", f"{mu:.2e} Pa·s")

        with tab2:
            df_mat = pd.DataFrame.from_dict(SOLID_MATERIALS, orient="index").drop("Personalizado")
            df_mat.columns = ["Condutividade k [W/(m·K)]", "Massa Específica ρ [kg/m³]", "Calor Específico cp [J/(kg·K)]"]
            st.dataframe(df_mat, use_container_width=True)

    elif menu == "Condução":
        st.title("🧱 Módulo de Condução Térmica")
        col_g, col_m = st.columns(2)
        with col_g:
            geom = st.selectbox("Geometria", ["Plana", "Cilíndrica", "Esférica"])
        with col_m:
            mat = st.selectbox("Material de Referência", list(SOLID_MATERIALS.keys()))

        k = st.number_input("Condutividade Térmica $k$ (W/m·K)", value=SOLID_MATERIALS[mat]["k"], format="%.4f")
        T1 = st.number_input("Temperatura Interna / Superfície 1 ($T_1$ em °C)", value=115.0)
        T2 = st.number_input("Temperatura Externa / Superfície 2 ($T_2$ em °C)", value=90.0)

        if geom == "Plana":
            L = st.number_input("Espessura da Parede $L$ (m)", value=0.05, min_value=1e-5, format="%.4f")
            A = st.number_input("Área Transversal $A$ (m²)", value=1.0, min_value=1e-5)
            res = ConductionSolver.plane_wall(k, A, L, T1, T2)
        elif geom == "Cilíndrica":
            L_cyl = st.number_input("Comprimento do Tubo $L$ (m)", value=40.0, min_value=1e-4)
            r_i = st.number_input("Raio Interno $r_i$ (m)", value=0.03, min_value=1e-5, format="%.4f")
            r_o = st.number_input("Raio Externo $r_o$ (m)", value=0.05, min_value=r_i + 1e-5, format="%.4f")
            res = ConductionSolver.cylindrical_wall(k, L_cyl, r_i, r_o, T1, T2)
        else:
            r_i = st.number_input("Raio Interno $r_i$ (m)", value=0.03, min_value=1e-5, format="%.4f")
            r_o = st.number_input("Raio Externo $r_o$ (m)", value=0.05, min_value=r_i + 1e-5, format="%.4f")
            res = ConductionSolver.spherical_wall(k, r_i, r_o, T1, T2)

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.success(f"**Taxa de Transferência de Calor ($q$):** `{res['q']:.2f} W`")
        c2.info(f"**Resistência Térmica ($R_t$):** `{res['R_t']:.5e} K/W`")

    elif menu == "Resistências Térmicas":
        st.title("⚡ Circuito Térmico e Parede Composta")
        st.markdown("Cálculo de fluxo térmico através de múltiplas camadas em série com perfil de temperatura.")

        A = st.number_input("Área da Seção Transversal $A$ (m²)", value=1.0, min_value=1e-4)
        c_t1, c_t2 = st.columns(2)
        T_inf1 = c_t1.number_input("Temperatura do Fluido Interno ($T_{\\infty,1}$ em °C)", value=150.0)
        h1 = c_t1.number_input("Coeficiente Convectivo Interno ($h_1$ em W/m²·K)", value=50.0)

        T_inf2 = c_t2.number_input("Temperatura do Fluido Externo ($T_{\\infty,2}$ em °C)", value=25.0)
        h2 = c_t2.number_input("Coeficiente Convectivo Externo ($h_2$ em W/m²·K)", value=15.0)

        n_layers = st.number_input("Número de Camadas Sólidas", min_value=1, max_value=5, value=2)

        R_conv1 = 1.0 / (h1 * A)
        R_conv2 = 1.0 / (h2 * A)
        layers = []

        for i in range(int(n_layers)):
            st.markdown(f"**Camada {i+1}**")
            col1, col2, col3 = st.columns(3)
            mat_i = col1.selectbox(f"Material {i+1}", list(SOLID_MATERIALS.keys()), index=min(i+5, len(SOLID_MATERIALS)-1), key=f"mat_{i}")
            L_i = col2.number_input(f"Espessura $L_{i+1}$ (m)", value=0.05, min_value=1e-5, format="%.4f", key=f"L_{i}")
            k_i = col3.number_input(f"Condutividade $k_{i+1}$ (W/m·K)", value=SOLID_MATERIALS[mat_i]["k"], min_value=1e-5, format="%.4f", key=f"k_{i}")
            layers.append({"name": f"Camada {i+1} ({mat_i})", "L": L_i, "k": k_i, "R": L_i / (k_i * A)})

        R_cond_tot = sum(l["R"] for l in layers)
        R_tot = R_conv1 + R_cond_tot + R_conv2
        q_net = (T_inf1 - T_inf2) / R_tot

        # Cálculo das temperaturas nas interfaces
        x_coords = [0.0]
        temps = [T_inf1 - q_net * R_conv1]
        curr_x = 0.0
        curr_T = temps[0]

        for l in layers:
            curr_x += l["L"]
            curr_T -= q_net * l["R"]
            x_coords.append(curr_x)
            temps.append(curr_T)

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("Resistência Térmica Total ($R_{tot}$)", f"{R_tot:.4f} K/W")
        c2.metric("Taxa de Calor Global ($q$)", f"{q_net:.2f} W")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_coords, y=temps, mode='lines+markers', name='Temperatura T(x)', line=dict(color='#ff4b4b', width=3)))
        fig.update_layout(title="Perfil de Temperatura Através das Camadas Sólidas", xaxis_title="Posição x (m)", yaxis_title="Temperatura (°C)")
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "Convecção Forçada Interna":
        st.title("🧪 Convecção Forçada Interna")
        col1, col2 = st.columns(2)
        fluid = col1.selectbox("Fluido", ["Água", "Ar"])
        process = col2.selectbox("Processo Térmico", ["Aquecimento do Fluido", "Resfriamento do Fluido"])

        D = st.number_input("Diâmetro Interno do Tubo $D$ (m)", value=0.05, min_value=1e-4)
        L = st.number_input("Comprimento do Tubo $L$ (m)", value=5.0, min_value=1e-3)
        V = st.number_input("Velocidade Média do Fluido $V$ (m/s)", value=1.5, min_value=1e-4)
        T_m = st.number_input("Temperatura Média do Fluido $T_m$ (°C)", value=50.0)

        res = ConvectionSolver.internal_pipe(D, L, V, T_m, fluid, heating=(process == "Aquecimento do Fluido"))

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reynolds ($Re_D$)", f"{res['Re']:.1f}")
        c2.metric("Regime", res["regime"])
        c3.metric("Nusselt ($Nu_D$)", f"{res['Nu']:.2f}")
        c4.metric("Coeficiente $h$", f"{res['h']:.2f} W/(m²·K)")

    elif menu == "Convecção Forçada Externa":
        st.title("💨 Convecção Forçada Externa (Placa Plana)")
        fluid = st.selectbox("Fluido", ["Ar", "Água"])
        L = st.number_input("Comprimento da Placa $L$ (m)", value=1.0, min_value=1e-4)
        V = st.number_input("Velocidade do Escoamento $V$ (m/s)", value=5.0, min_value=1e-4)
        Ts = st.number_input("Temperatura da Superfície $T_s$ (°C)", value=80.0)
        Tinf = st.number_input("Temperatura da Corrente Livre $T_\\infty$ (°C)", value=20.0)

        res = ConvectionSolver.external_flat_plate(L, V, Ts, Tinf, fluid)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Reynolds ($Re_L$)", f"{res['Re']:.2e}")
        c2.metric("Regime", res['regime'])
        c3.metric("Coeficiente $h$", f"{res['h']:.2f} W/(m²·K)")
        st.success(f"**Fluxo Térmico Médio ($q''$):** `{res['flux']:.2f} W/m²` (Avaliado a $T_f = {res['T_film']:.1f} °C$)")

    elif menu == "Convecção Natural":
        st.title("🌡️ Convecção Natural (Placa Vertical)")
        fluid = st.selectbox("Fluido", ["Ar", "Água"])
        L = st.number_input("Altura da Placa $L$ (m)", value=0.5, min_value=1e-4)
        Ts = st.number_input("Temperatura da Superfície $T_s$ (°C)", value=60.0)
        Tinf = st.number_input("Temperatura do Fluido $T_\\infty$ (°C)", value=20.0)

        res = ConvectionSolver.natural_convection_vertical_plate(L, Ts, Tinf, fluid)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Grashof ($Gr_L$)", f"{res['Gr']:.2e}")
        c2.metric("Rayleigh ($Ra_L$)", f"{res['Ra']:.2e}")
        c3.metric("Regime", res['regime'])
        st.success(f"**Coeficiente Convectivo ($h$):** `{res['h']:.2f} W/(m²·K)` | **Fluxo ($q''$):** `{res['flux']:.2f} W/m²`")

    elif menu == "Aletas":
        st.title("📌 Módulo de Superfícies Estendidas (Aletas)")
        st.markdown("Análise unidimensional para aleta retangular de seção transversal uniforme (Incropera, Cap. 3).")

        col_bc, col_mat = st.columns(2)
        with col_bc:
            bc_type = st.selectbox(
                "Condição de Contorno na Extremidade ($x = L$)",
                ["Extremidade Convectiva", "Extremidade Adiabática", "Temperatura Prescrita na Ponta", "Aleta Infinita"]
            )
        with col_mat:
            mat = st.selectbox("Material da Aleta", list(SOLID_MATERIALS.keys()), index=2)

        c1, c2, c3 = st.columns(3)
        k = c1.number_input("Condutividade $k$ (W/m·K)", value=SOLID_MATERIALS[mat]["k"], min_value=0.01)
        t = c2.number_input("Espessura $t$ (m)", value=0.005, min_value=1e-5, format="%.4f")
        w = c3.number_input("Largura $w$ (m)", value=0.10, min_value=1e-4, format="%.4f")

        c4, c5, c6 = st.columns(3)
        L = c4.number_input("Comprimento $L$ (m)", value=0.05, min_value=1e-4, format="%.4f")
        h = c5.number_input("Coeficiente Convectivo $h$ (W/m²·K)", value=25.0, min_value=0.1)
        Tb = c6.number_input("Temperatura na Base $T_b$ (°C)", value=100.0)

        c7, c8 = st.columns(2)
        Tinf = c7.number_input("Temperatura do Ambiente $T_\\infty$ (°C)", value=25.0)
        TL = c8.number_input("Temperatura na Ponta $T_L$ (°C)", value=40.0, disabled=(bc_type != "Temperatura Prescrita na Ponta"))

        res = FinSolver.rectangular_fin(k, t, w, L, h, Tb, Tinf, bc_type, TL)

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Taxa de Calor ($q_f$)", f"{res['q_fin']:.2f} W")
        m2.metric("Eficiência ($\\eta_f$)", f"{res['eta']*100:.1f} %")
        m3.metric("Efetividade ($\\varepsilon_f$)", f"{res['effectiveness']:.2f}")
        m4.metric("Parâmetro $m$", f"{res['m']:.2f} m⁻¹")

        if res['effectiveness'] < 2.0:
            st.warning("⚠️ **Alerta de Projeto:** Efetividade ($\\varepsilon_f < 2$). O uso desta aleta não se justifica economicamente; considere aumentar a razão $P/A_c$ ou utilizar material de maior condutividade.")

        df_fin = pd.DataFrame({"Posição x (mm)": res['x_vals'] * 1000.0, "Temperatura (°C)": res['T_vals']})
        fig = px.line(df_fin, x="Posição x (mm)", y="Temperatura (°C)", title=f"Perfil de Temperatura ao Longo da Aleta ({bc_type})")
        fig.add_hline(y=Tinf, line_dash="dash", line_color="gray", annotation_text="T_inf")
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("📥 Baixar Dados da Aleta (.csv)", df_fin.to_csv(index=False).encode("utf-8"), "perfil_aleta.csv", "text/csv")

    elif menu == "Regime Transiente":
        st.title("⏱️ Análise em Regime Transiente")
        st.markdown("Método da Capacitância Global (Sistemas Concentrados).")

        mat = st.selectbox("Material do Corpo", list(SOLID_MATERIALS.keys()), index=3)
        c1, c2, c3 = st.columns(3)
        rho = c1.number_input("Massa Específica $\\rho$ (kg/m³)", value=SOLID_MATERIALS[mat]["rho"])
        cp = c2.number_input("Calor Específico $c_p$ (J/kg·K)", value=SOLID_MATERIALS[mat]["cp"])
        k = c3.number_input("Condutividade Térmica $k$ (W/m·K)", value=SOLID_MATERIALS[mat]["k"])

        c4, c5, c6 = st.columns(3)
        V = c4.number_input("Volume $V$ (m³)", value=0.001, format="%.5f")
        As = c5.number_input("Área Superficial $A_s$ (m²)", value=0.10, format="%.4f")
        h = c6.number_input("Coeficiente $h$ (W/m²·K)", value=50.0)

        c7, c8, c9 = st.columns(3)
        Ti = c7.number_input("Temperatura Inicial $T_i$ (°C)", value=300.0)
        Tinf = c8.number_input("Temperatura do Fluido $T_\\infty$ (°C)", value=25.0)
        t_final = c9.number_input("Tempo Total (s)", value=600.0)

        df, biot, tau = TransientSolver.lumped_capacitance(rho, V, cp, As, h, k, Ti, Tinf, t_final)

        st.markdown("---")
        m1, m2 = st.columns(2)
        m1.metric("Número de Biot ($Bi = h L_c / k$)", f"{biot:.4f}", help="Critério de validade: Bi < 0.1")
        m2.metric("Constante de Tempo Térmica ($\\tau_t$)", f"{tau:.1f} s")

        if biot > 0.1:
            st.warning("⚠️ **Atenção:** $Bi > 0.1$. Os gradientes internos de temperatura não são desprezíveis; recomenda-se usar aproximação de um termo (Cartas de Heisler).")

        fig = px.line(df, x="Tempo (s)", y="Temperatura (°C)", title="Evolução Térmica Transiente")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("📥 Baixar Curva Transiente (.csv)", df.to_csv(index=False).encode("utf-8"), "transiente.csv", "text/csv")

    elif menu == "Trocadores de Calor":
        st.title("🔄 Projeto e Análise de Trocadores de Calor")
        tab_lmtd, tab_ntu = st.tabs(["Método MLDT (Diferença Média Logarítmica)", "Método ε-NTU (Efetividade)"])

        with tab_lmtd:
            flow = st.selectbox("Arranjo do Escoamento", ["Contracorrente", "Correntes Paralelas"])
            c1, c2 = st.columns(2)
            Thi = c1.number_input("Entrada Fluido Quente $T_{h,i}$ (°C)", value=90.0)
            Tho = c1.number_input("Saída Fluido Quente $T_{h,o}$ (°C)", value=50.0)
            Tci = c2.number_input("Entrada Fluido Frio $T_{c,i}$ (°C)", value=20.0)
            Tco = c2.number_input("Saída Fluido Frio $T_{c,o}$ (°C)", value=40.0)

            res = HeatExchangerSolver.lmtd_analysis(Thi, Tho, Tci, Tco, flow)
            st.markdown("---")
            m1, m2 = st.columns(2)
            m1.metric("MLDT ($\\Delta T_{ml}$)", f"{res['LMTD']:.2f} °C")
            m2.metric("$\\Delta T_1$ / $\\Delta T_2$", f"{res['DT1']:.1f} °C / {res['DT2']:.1f} °C")

        with tab_ntu:
            config = st.selectbox("Configuração do Trocador", ["Contracorrente", "Correntes Paralelas", "Casco e Tubos (1 passe casco)"])
            c1, c2, c3 = st.columns(3)
            Ch = c1.number_input("Capacidade Calorífica Quente $C_h = \\dot{m}_h c_{p,h}$ (W/K)", value=4000.0)
            Cc = c2.number_input("Capacidade Calorífica Fria $C_c = \\dot{m}_c c_{p,c}$ (W/K)", value=2500.0)
            UA = c3.number_input("Condutância Global $U \\cdot A$ (W/K)", value=3500.0)

            c4, c5 = st.columns(2)
            Thi_n = c4.number_input("Temperatura Entrada Quente $T_{h,i}$ (°C)", value=120.0, key="ntu_thi")
            Tci_n = c5.number_input("Temperatura Entrada Fria $T_{c,i}$ (°C)", value=20.0, key="ntu_tci")

            res_ntu = HeatExchangerSolver.ntu_analysis(Ch, Cc, UA, Thi_n, Tci_n, config)
            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Efetividade ($\\varepsilon$)", f"{res_ntu['eps']*100:.1f} %")
            m2.metric("NTU", f"{res_ntu['NTU']:.2f}")
            m3.metric("Calor Trocado ($q$)", f"{res_ntu['q']/1000:.2f} kW")
            m4.metric("Razão $C_r$", f"{res_ntu['Cr']:.2f}")

            st.info(f"**Temperaturas de Saída Calculadas:** $T_{{h,o}} = {res_ntu['Tho']:.2f} °C$ | $T_{{c,o}} = {res_ntu['Tco']:.2f} °C$")

    elif menu == "Radiação Térmica":
        st.title("☀️ Módulo de Radiação Térmica")
        st.markdown("Troca radiativa líquida entre uma superfície cinzenta difusa e uma vizinhança isotérmica grande.")

        c1, c2 = st.columns(2)
        eps = c1.slider("Emissividade da Superfície ($\\varepsilon$)", 0.01, 1.00, 0.85)
        A = c2.number_input("Área da Superfície $A$ (m²)", value=1.5, min_value=1e-4)

        c3, c4 = st.columns(2)
        Ts = c3.number_input("Temperatura da Superfície $T_s$ (°C)", value=250.0)
        Tsur = c4.number_input("Temperatura da Vizinhança $T_{viz}$ (°C)", value=25.0)

        res = RadiationSolver.surface_surroundings(eps, A, Ts, Tsur)
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Taxa Líquida de Radiação ($q_{rad}$)", f"{res['q_rad']:.2f} W")
        m2.metric("Poder Emissivo ($E$)", f"{res['E']:.1f} W/m²")
        m3.metric("Coeficiente Radiativo ($h_r$)", f"{res['h_rad']:.2f} W/(m²·K)")

    elif menu == "Resolver Exercício":
        st.title("🤖 Solucionador Automático de Enunciados")
        user_prompt = st.text_area("Cole o enunciado do problema:", placeholder="Ex: Um trocador contracorrente opera com temperaturas de 90°C, 50°C, 20°C e 40°C...")
        if st.button("Analisar e Resolver"):
            if user_prompt.strip():
                st.markdown(AutoSolver.parse_and_solve(user_prompt))
            else:
                st.error("Por favor, insira um enunciado válido.")

    elif menu == "Sobre":
        st.title("ℹ️ Sobre o Heat Transfer Studio")
        st.markdown("""
        **Heat Transfer Studio (BETA)** — Ferramenta computacional para apoio acadêmico e projetos de engenharia térmica.
        - **Disciplina:** Fenômenos de Transporte II (LOQ 4084)
        - **Instituição:** Escola de Engenharia de Lorena — Universidade de São Paulo (EEL-USP)
        - **Referência Bibliográfica Base:** Incropera, F. P. et al. *Fundamentos de Transferência de Calor e de Massa*.
        """)

if __name__ == "__main__":
    main()
