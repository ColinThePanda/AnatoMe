from dataclasses import dataclass, asdict
from typing import cast
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

@dataclass
class Params:
    """Dataclass for input params for the simulation

    Raises:
        ValueError: If duration is <= 0
        ValueError: If drinks is < 0
        ValueError: If food eaten is < 0
        ValueError: If body mass is <= 0
        ValueError: If the matabolism preset is <= 0
        ValueError: If aldh efficiency is <= 0
        ValueError: If body water fraction is <= 0
        ValueError: If total time is <= 0
    """
    drinks: float = 3 # standard drinks = 14g ethanol
    duration: float = 2 # how long drinking
    food_eaten: float = 250 # g
    body_mass: float = 70 # kg
    metab_preset: float = 1.0 # from dropdown on site
    aldh_efficiency: float = 1.0 # from dropdown on site
    body_water_fraction: float = 0.68 # %
    total_time: float = 12 # how long to simulate

    def __post_init__(self):
        if self.duration <= 0:
            raise ValueError("duration must be greater than 0")
        if self.drinks < 0:
            raise ValueError("drinks cannot be negative")
        if self.food_eaten < 0:
            raise ValueError("food_eaten cannot be negative")
        if self.body_mass <= 0:
            raise ValueError("body_mass must be greater than 0")
        if self.metab_preset <= 0:
            raise ValueError("metab_preset must be greater than 0")
        if self.aldh_efficiency <= 0:
            raise ValueError("aldh_efficiency must be greater than 0")
        if self.body_water_fraction <= 0:
            raise ValueError("body_water_fraction must be greater than 0")
        if self.total_time <= 0:
            raise ValueError("total_time must be greater than 0")

@dataclass
class SimulationData:
    """Data class for the data produced by the simulation"""
    time_hrs: float
    stomach_ethanol_g: float
    intestine_ethanol_g: float
    body_ethanol_g: float
    acetaldehyde_g: float
    acetate_g: float
    ethanol_concentration_mmol_l: float
    acetaldehyde_concentration_mmol_l: float
    input_g_h: float
    ethanol_elimination_g_h: float
    acetaldehyde_elimination_g_h: float
    ethanol_absorption_g_h: float
    stomach_to_body_g_h: float
    intestine_to_body_g_h: float
    stomach_to_intestine_g_h: float
    bac_percent: float

class Simulation:
    """The alcohol absorption and metabolism simulation"""
    def __init__(self, params: Params | None = None):
        self.sim_rate: float = 1 / 600 # constant for accuracy
        self.params = params if params is not None else Params()
        self.time_hrs = 0.0
        self.stomach_ethanol_g = 0.0
        self.intestine_ethanol_g = 0.0
        self.body_ethanol_g = 0.0
        self.acetaldehyde_g = 0.0
        self.acetate_g = 0.0
        self.sim_data: list[SimulationData] = []

    def step(self):
        """Steps 1 tick in the simulation"""
        sr = self.sim_rate
        S = self.stomach_ethanol_g
        I = self.intestine_ethanol_g
        B = self.body_ethanol_g
        H = self.acetaldehyde_g
        Ac = self.acetate_g

        # Most of these equations are derived from science papers so they look messy and have constants that are just from real data, not magic numbers
        dose_g = 14 * self.params.drinks
        input_g_h = dose_g / self.params.duration if self.time_hrs < self.params.duration else 0
        food_factor = self.params.food_eaten / (self.params.food_eaten + 500)
        k_empty_base_h = 4.2
        k_empty_h = k_empty_base_h / (1 + 4.5 * food_factor)
        kS_h = 0.22
        Ka_base_h = 7.0
        Ka_h = Ka_base_h / (1 + 1.0 * food_factor)

        Vd_l = self.params.body_water_fraction * self.params.body_mass
        C_ethanol = 1000 * B / (46.068 * Vd_l)
        Vmax_ethanol = 3.256 * self.params.metab_preset
        E_g_h = (Vmax_ethanol * C_ethanol / (0.8183 + C_ethanol)) * (46.068 / 1000) * 60 if C_ethanol > 0 else 0
        E_g_h = min(E_g_h, B / sr)

        C_acetaldehyde = 1000 * H / (44.053 * Vd_l)
        liver_mass_kg = 0.026 * self.params.body_mass
        Vmax_acetaldehyde = 2.7 * liver_mass_kg * self.params.aldh_efficiency
        EH_g_h = (Vmax_acetaldehyde * C_acetaldehyde / (0.0012 + C_acetaldehyde)) * (44.053 / 1000) * 60 if C_acetaldehyde > 0 else 0
        EH_g_h = min(EH_g_h, H / sr)
        
        stomach_to_body_g_h = kS_h * S
        stomach_to_intestine_g_h = k_empty_h * S
        intestine_to_body_g_h = Ka_h * I
        ethanol_absorption_g_h = stomach_to_body_g_h + intestine_to_body_g_h

        dS_g_h = input_g_h - kS_h * S - k_empty_h * S
        dI_g_h = k_empty_h * S - Ka_h * I
        dB_g_h = ethanol_absorption_g_h - E_g_h
        dH_g_h = E_g_h * (44.053 / 46.068) - EH_g_h
        dAc_g_h = EH_g_h * (59.044 / 44.053)

        # Cap the ethanol in each compartment to 0
        self.stomach_ethanol_g = max(S + sr * dS_g_h, 0)
        self.intestine_ethanol_g = max(I + sr * dI_g_h, 0)
        self.body_ethanol_g = max(B + sr * dB_g_h, 0)
        self.acetaldehyde_g = max(H + sr * dH_g_h, 0)
        self.acetate_g = max(Ac + sr * dAc_g_h, 0)

        # recalculate values for accurate sim data
        C_ethanol = 1000 * self.body_ethanol_g / (46.068 * Vd_l)
        C_acetaldehyde = 1000 * self.acetaldehyde_g / (44.053 * Vd_l)
        bac_percent = self.body_ethanol_g / (self.params.body_water_fraction * self.params.body_mass * 10)

        self.sim_data.append(SimulationData(self.time_hrs,
                                            self.stomach_ethanol_g,
                                            self.intestine_ethanol_g,
                                            self.body_ethanol_g,
                                            self.acetaldehyde_g,
                                            self.acetate_g, C_ethanol,
                                            C_acetaldehyde,
                                            input_g_h,
                                            E_g_h,
                                            EH_g_h,
                                            ethanol_absorption_g_h,
                                            stomach_to_body_g_h,
                                            intestine_to_body_g_h,
                                            stomach_to_intestine_g_h, bac_percent))

    def simulate(self):
        """Simulates for the given total time in the parameters"""
        while self.time_hrs < self.params.total_time:
            self.step()
            self.time_hrs += self.sim_rate

def main() -> None:
    """Main function"""
    st.set_page_config(page_title="BAC Simulation", layout="wide")

    st.title("Blood Alcohol Simulation")

    st.sidebar.header("Simulation Parameters")

    metab_values: dict[str, float] = {
        "Slow": 0.75,
        "Average": 1.0,
        "Fast": 1.25,
    }

    aldh_values: dict[str, float] = {
        "Normal": 1.0,
        "Reduced": 0.7,
        "Very reduced": 0.5,
    }

    if "results" not in st.session_state:
        st.session_state["results"] = None

    with st.sidebar.form("simulation_form"):
        drinks: int = int(st.slider("Drinks consumed", 0, 12, 3, 1))
        duration: float = float(st.slider("Drinking duration (hours)", 0.1, 8.0, 2.0, 0.1, format="%.1f"))
        food_eaten: int = int(st.slider("Food eaten (grams)", 0, 1500, 250, 50))
        body_mass: int = int(st.slider("Body mass (kg)", 30, 150, 70, 1))
        body_water_fraction: int = int(st.slider("Body Water Fraction (%)", 50, 75, 68, 1))
        total_time: int = int(st.slider("Simulation time (hours)", 2, 32, 12, 1))

        metab_choice: str = str(st.selectbox("Ethanol metabolism preset", ["Slow", "Average", "Fast"], index=1))
        aldh_choice: str = str(st.selectbox("ALDH efficiency", ["Normal", "Reduced", "Very reduced"], index=0))

        run: bool = bool(st.form_submit_button("Run simulation"))

    if run:
        params: Params = Params(
            drinks=drinks,
            duration=duration,
            food_eaten=food_eaten,
            body_mass=body_mass,
            metab_preset=metab_values[metab_choice],
            aldh_efficiency=aldh_values[aldh_choice],
            body_water_fraction=body_water_fraction/100,
            total_time=total_time,
        )

        sim: Simulation = Simulation(params)
        sim.simulate()

        df: pd.DataFrame = pd.DataFrame([asdict(point) for point in sim.sim_data]) # easier to graph and do operations on every data point
        peak: SimulationData = max(sim.sim_data, key=lambda point: point.bac_percent) # max BAC
        final: SimulationData = sim.sim_data[-1]

        drink_times: list[float] = []
        if drinks == 1:
            drink_times = [0.0]
        elif drinks > 1:
            drink_times = [i * duration / (drinks - 1) for i in range(drinks)]

        st.session_state["results"] = {
            "df": df,
            "peak": peak,
            "final": final,
            "drink_times": drink_times,
            "total_time": total_time,
            "params": params,
            "metab_choice": metab_choice,
            "aldh_choice": aldh_choice,
            "sim_rate": sim.sim_rate,
        }

    if st.session_state["results"] is None:
        return

    results: dict[str, object] = cast(dict[str, object], st.session_state["results"])

    df: pd.DataFrame = cast(pd.DataFrame, results["df"])
    peak: SimulationData = cast(SimulationData, results["peak"])
    final: SimulationData = cast(SimulationData, results["final"])
    drink_times: list[float] = cast(list[float], results["drink_times"])
    displayed_total_time: int = cast(int, results["total_time"])
    displayed_params: Params = cast(Params, results["params"])
    displayed_metab_choice: str = cast(str, results["metab_choice"])
    displayed_aldh_choice: str = cast(str, results["aldh_choice"])
    sim_rate: float = cast(float, results["sim_rate"])

    current_settings: dict[str, object] = {
        "drinks": drinks,
        "duration": duration,
        "food_eaten": food_eaten,
        "body_mass": body_mass,
        "total_time": total_time,
        "metab_choice": metab_choice,
        "aldh_choice": aldh_choice,
    }

    last_run_settings: dict[str, object] = {
        "drinks": displayed_params.drinks,
        "duration": displayed_params.duration,
        "food_eaten": displayed_params.food_eaten,
        "body_mass": displayed_params.body_mass,
        "total_time": displayed_params.total_time,
        "metab_choice": displayed_metab_choice,
        "aldh_choice": displayed_aldh_choice,
    }

    # display if params changed but did not run
    if current_settings != last_run_settings:
        st.caption(
            f"Showing last run: {displayed_params.drinks:g} drinks, "
            f"{displayed_params.duration:.1f} h duration, "
            f"{displayed_params.food_eaten:g} g food, "
            f"{displayed_params.body_mass:g} kg, "
            f"{displayed_params.total_time:g} h simulation, "
            f"{displayed_metab_choice} metabolism, "
            f"{displayed_aldh_choice} ALDH."
        )

    def add_drink_lines(ax: Axes) -> None:
        """Add dashed lines for each drink in black and a red dashed line for the last drink

        Args:
            ax (Axes): Graph to add the lines to
        """
        for i, drink_time in enumerate(drink_times):
            if i == len(drink_times) - 1:
                ax.axvline(drink_time, color="red", linestyle="--", linewidth=1.6, label="Last drink")
            elif i == 0:
                ax.axvline(drink_time, color="black", linestyle="--", linewidth=1.2, alpha=0.9, label="Drink")
            else:
                ax.axvline(drink_time, color="black", linestyle="--", linewidth=1.2, alpha=0.9)

    def threshold_interval(threshold: float) -> str:
        """Gets the start time, end time, and duration in which the blood alcohol content is over a certain threshold

        Args:
            threshold (float): Threshold to get times when the blood alcohol content is over

        Returns:
            str: Formatted text of the start, end, and duration in which the blood alcohol content is greater than the threshold
        """
        above = df[df["bac_percent"] >= threshold]["time_hrs"]
        if len(above) == 0:
            return "Never"

        start: float = float(above.iloc[0])
        end: float = float(above.iloc[-1])
        duration_at: float = end - start

        return f"{start:.2f}-{end:.2f} h ({duration_at:.2f} h)"

    def time_when_alcohol_fully_absorbed() -> str:
        """Gets the time when the alcohol is fully absorbed (gut ethanol <= 0.01g)

        Returns:
            str: The text "Not reached" or a formatted sring of the time
        """
        gut_ethanol = df["stomach_ethanol_g"] + df["intestine_ethanol_g"]
        after_drinking = df[df["time_hrs"] >= displayed_params.duration].copy()
        after_drinking["gut_ethanol_g"] = gut_ethanol[df["time_hrs"] >= displayed_params.duration]

        absorbed = after_drinking[after_drinking["gut_ethanol_g"] <= 0.01]

        if len(absorbed) == 0:
            return "Not reached"

        time_hrs: float = float(absorbed.iloc[0]["time_hrs"])
        return f"{time_hrs:.2f} h"

    def time_when_alcohol_fully_metabolized() -> str:
        """Gets the time when the alcohol is fully metabolized (body ethanol <= 0.01g)

        Returns:
            str: The text "Not reached" or a formatted sring of the time
        """
        after_peak = df[df["time_hrs"] >= peak.time_hrs]
        metabolized = after_peak[after_peak["body_ethanol_g"] <= 0.01]

        if len(metabolized) == 0:
            return "Not reached"

        time_hrs: float = float(metabolized.iloc[0]["time_hrs"])
        return f"{time_hrs:.2f} h"

    auc_bac: float = float((df["bac_percent"] * sim_rate).sum()) # area under the BAC curve, BAC exposure

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Peak BAC", f"{peak.bac_percent:.3f}%")
    col2.metric("Time to peak", f"{peak.time_hrs:.2f} h")
    col3.metric("Time at 0.08%", threshold_interval(0.08))
    col4.metric("BAC exposure", f"{auc_bac:.3f} %-h")

    st.subheader("Graphs")

    graph_col1, graph_col2 = st.columns(2)
    graph_col3, graph_col4 = st.columns(2)

    with graph_col1:
        st.write("BAC Over Time")

        fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
        ax.plot(df["time_hrs"], df["bac_percent"], label="BAC", linewidth=2)

        add_drink_lines(ax)

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("BAC (%)")
        ax.set_title("BAC Over Time")
        ax.set_xlim(0, displayed_total_time)
        ax.grid(True)
        ax.legend(fontsize=7)

        st.pyplot(fig, use_container_width=True)

    with graph_col2:
        st.write("Compartment Amounts")

        fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
        ax.plot(df["time_hrs"], df["stomach_ethanol_g"], label="Stomach ethanol", linewidth=2)
        ax.plot(df["time_hrs"], df["intestine_ethanol_g"], label="Intestine ethanol", linewidth=2)
        ax.plot(df["time_hrs"], df["body_ethanol_g"], label="Body ethanol", linewidth=2)
        ax.plot(df["time_hrs"], df["acetaldehyde_g"], label="Acetaldehyde", linewidth=2)
        ax.plot(df["time_hrs"], df["acetate_g"], label="Total acetate", linewidth=2)

        add_drink_lines(ax)

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Amount (g)")
        ax.set_title("Compartments")
        ax.set_xlim(0, displayed_total_time)
        ax.grid(True)
        ax.legend(fontsize=7)

        st.pyplot(fig, use_container_width=True)

    with graph_col3:
        st.write("Absorption and Elimination Rates")

        fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)

        ax.plot(df["time_hrs"], df["ethanol_absorption_g_h"], label="Ethanol absorption", linewidth=2)
        ax.plot(df["time_hrs"], df["ethanol_elimination_g_h"], label="Ethanol elimination", linewidth=2)
        ax.plot(df["time_hrs"], df["acetaldehyde_elimination_g_h"], label="Acetaldehyde elimination", linewidth=2)

        add_drink_lines(ax)

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Rate (g/h)")
        ax.set_title("Absorption and Elimination Rates")
        ax.set_xlim(0, displayed_total_time)
        ax.grid(True)
        ax.legend(fontsize=7)

        st.pyplot(fig, use_container_width=True)

    with graph_col4:
        st.write("Acetaldehyde Over Time")

        fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)

        ax.plot(df["time_hrs"], df["acetaldehyde_g"], label="Acetaldehyde amount", linewidth=2)

        add_drink_lines(ax)

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Acetaldehyde (g)")
        ax.set_title("Acetaldehyde Over Time")
        ax.set_xlim(0, displayed_total_time)
        ax.grid(True)
        ax.legend(fontsize=7)

        st.pyplot(fig, use_container_width=True)

    st.subheader("Extra Stats")

    total_absorbed_g: float = float((df["ethanol_absorption_g_h"] * sim_rate).sum())
    total_metabolized_g: float = float((df["ethanol_elimination_g_h"] * sim_rate).sum())
    max_absorption_rate: float = float(df["ethanol_absorption_g_h"].max())
    max_elimination_rate: float = float(df["ethanol_elimination_g_h"].max())

    peak_acetaldehyde = df.loc[df["acetaldehyde_g"].idxmax()]
    peak_acetaldehyde_g: float = float(peak_acetaldehyde["acetaldehyde_g"]) # type: ignore
    peak_acetaldehyde_time: float = float(peak_acetaldehyde["time_hrs"]) # type: ignore

    acetaldehyde_exposure: float = float((df["acetaldehyde_concentration_mmol_l"] * sim_rate).sum())

    alcohol_fully_absorbed_text: str = time_when_alcohol_fully_absorbed()
    alcohol_fully_metabolized_text: str = time_when_alcohol_fully_metabolized()

    bac_timing_stats: pd.DataFrame = pd.DataFrame([
        {"Stat": "Time at 0.08% BAC", "Value": threshold_interval(0.08)},
        {"Stat": "Time at 0.05% BAC", "Value": threshold_interval(0.05)},
        {"Stat": "Time at 0.02% BAC", "Value": threshold_interval(0.02)},
        {"Stat": "BAC exposure", "Value": f"{auc_bac:.3f} %-h"},
    ])

    ethanol_stats: pd.DataFrame = pd.DataFrame([
        {"Stat": "Total ethanol absorbed", "Value": f"{total_absorbed_g:.2f} g"},
        {"Stat": "Total ethanol metabolized", "Value": f"{total_metabolized_g:.2f} g"},
        {"Stat": "Alcohol fully absorbed", "Value": alcohol_fully_absorbed_text},
        {"Stat": "Alcohol fully metabolized", "Value": alcohol_fully_metabolized_text},
        {"Stat": "Max absorption rate", "Value": f"{max_absorption_rate:.2f} g/h"},
        {"Stat": "Max elimination rate", "Value": f"{max_elimination_rate:.2f} g/h"},
    ])

    metabolism_stats: pd.DataFrame = pd.DataFrame([
        {"Stat": "Peak acetaldehyde", "Value": f"{peak_acetaldehyde_g:.4f} g at {peak_acetaldehyde_time:.2f} h"},
        {"Stat": "Acetaldehyde exposure", "Value": f"{acetaldehyde_exposure:.4f} mmol/L·h"},
        {"Stat": "Final body ethanol", "Value": f"{final.body_ethanol_g:.2f} g"},
        {"Stat": "Final acetate", "Value": f"{final.acetate_g:.2f} g"},
    ])

    stat_col1, stat_col2, stat_col3 = st.columns(3)

    with stat_col1:
        st.markdown("#### BAC Timing")
        st.dataframe(bac_timing_stats, hide_index=True, use_container_width=True)

    with stat_col2:
        st.markdown("#### Ethanol Movement")
        st.dataframe(ethanol_stats, hide_index=True, use_container_width=True)

    with stat_col3:
        st.markdown("#### Metabolism")
        st.dataframe(metabolism_stats, hide_index=True, use_container_width=True)

    with st.expander("Show raw data"):
        st.dataframe(df)

if __name__ == "__main__":
    main()
