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
        # Called after object is created
        # Raise errors if the inputs are invalid
        if self.duration <= 0:
            raise ValueError("duration must be greater than 0") # duration > 0
        if self.drinks < 0:
            raise ValueError("drinks cannot be negative") # drinks < 0
        if self.food_eaten < 0:
            raise ValueError("food_eaten cannot be negative") # food eaten < 0
        if self.body_mass <= 0:
            raise ValueError("body_mass must be greater than 0") # body mass <= 0
        if self.metab_preset <= 0:
            raise ValueError("metab_preset must be greater than 0") # metabolism preset <= 0
        if self.aldh_efficiency <= 0:
            raise ValueError("aldh_efficiency must be greater than 0") # ALDH efficiency <= 0
        if self.body_water_fraction <= 0:
            raise ValueError("body_water_fraction must be greater than 0") # body water fraction <= 0
        if self.total_time <= 0:
            raise ValueError("total_time must be greater than 0") # total time <= 0

@dataclass
class SimulationData:
    """Data class for the data produced by the simulation"""
    time_hrs: float # time in hours
    stomach_ethanol_g: float # stomach ethanol in grams
    intestine_ethanol_g: float # intestine ethanol in grams
    body_ethanol_g: float # body ethanol in grams
    acetaldehyde_g: float # acetaldehyde in grams
    acetate_g: float # acetate in grams
    ethanol_concentration_mmol_l: float # ethanol concentratio in millimoles per liter
    acetaldehyde_concentration_mmol_l: float # acetaldehyde concentration in millimoles per liter
    input_g_h: float # input ethanol in grams per hour
    ethanol_elimination_g_h: float # ethanol elimination rate in grams per hour
    acetaldehyde_elimination_g_h: float # acetaldehyde elimination rate in grams per hour
    ethanol_absorption_g_h: float # ethanol absorption rate in grams per hour
    stomach_to_body_g_h: float # rate of transfer of ethanol from stomach to body in grams per hour
    intestine_to_body_g_h: float # rate of transfer of ethanol from intestine to body in grams per hour
    stomach_to_intestine_g_h: float # rate of transfer of ethanol from stomach to intestine
    bac_percent: float # Blood Alcohol Concentration (BAC) percent in the body

class Simulation:
    """The alcohol absorption and metabolism simulation"""
    def __init__(self, params: Params | None = None):
        self.sim_rate: float = 1 / 600 # constant for accuracy, 1/3mins
        self.params = params if params is not None else Params() # initialize default params if not given
        self.time_hrs = 0.0 # initialize time to 0
        self.stomach_ethanol_g = 0.0 # initialize stomach ethanol to 0
        self.intestine_ethanol_g = 0.0 # initialize intestine ethanol to 0
        self.body_ethanol_g = 0.0 # initialize body ethanol to 0
        self.acetaldehyde_g = 0.0 # initialize acetaldehyde to 0
        self.acetate_g = 0.0 # initialize acetate to 0
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
        dose_g = 14 * self.params.drinks # 1 standard drink = 14g ethanol
        input_g_h = dose_g / self.params.duration if self.time_hrs < self.params.duration else 0 # input ethanol per hour = total ethanol dose / drinking time
        food_factor = self.params.food_eaten / (self.params.food_eaten + 500) # saturating function (approximation)
        
        # equations from Sadighi paper (food)
        k_empty_base_h = 4.2
        k_empty_h = k_empty_base_h / (1 + 4.5 * food_factor)
        kS_h = 0.22
        Ka_base_h = 7.0
        Ka_h = Ka_base_h / (1 + 1.0 * food_factor)

        # equations from Lee paper (absorption)
        Vd_l = self.params.body_water_fraction * self.params.body_mass
        C_ethanol = 1000 * B / (46.068 * Vd_l)
        Vmax_ethanol = 3.256 * self.params.metab_preset
        E_g_h = (Vmax_ethanol * C_ethanol / (0.8183 + C_ethanol)) * (46.068 / 1000) * 60 if C_ethanol > 0 else 0
        E_g_h = min(E_g_h, B / sr)

        # equations from Umulis paper (acetaldehyde)
        C_acetaldehyde = 1000 * H / (44.053 * Vd_l)
        liver_mass_kg = 0.026 * self.params.body_mass
        Vmax_acetaldehyde = 2.7 * liver_mass_kg * self.params.aldh_efficiency
        EH_g_h = (Vmax_acetaldehyde * C_acetaldehyde / (0.0012 + C_acetaldehyde)) * (44.053 / 1000) * 60 if C_acetaldehyde > 0 else 0
        EH_g_h = min(EH_g_h, H / sr)
        
        # equations from Lee paper (ethanol compartment movement)
        stomach_to_body_g_h = kS_h * S
        stomach_to_intestine_g_h = k_empty_h * S
        intestine_to_body_g_h = Ka_h * I
        ethanol_absorption_g_h = stomach_to_body_g_h + intestine_to_body_g_h
        
        # compartment ethanol rate per hour
        dS_g_h = input_g_h - kS_h * S - k_empty_h * S
        dI_g_h = k_empty_h * S - Ka_h * I
        dB_g_h = ethanol_absorption_g_h - E_g_h
        dH_g_h = E_g_h * (44.053 / 46.068) - EH_g_h
        dAc_g_h = EH_g_h * (59.044 / 44.053)

        # multiply rates per hour by delta time and cap to 0
        self.stomach_ethanol_g = max(S + sr * dS_g_h, 0)
        self.intestine_ethanol_g = max(I + sr * dI_g_h, 0)
        self.body_ethanol_g = max(B + sr * dB_g_h, 0)
        self.acetaldehyde_g = max(H + sr * dH_g_h, 0)
        self.acetate_g = max(Ac + sr * dAc_g_h, 0)

        # recalculate values for accurate sim data
        C_ethanol = 1000 * self.body_ethanol_g / (46.068 * Vd_l)
        C_acetaldehyde = 1000 * self.acetaldehyde_g / (44.053 * Vd_l)
        bac_percent = self.body_ethanol_g / (self.params.body_water_fraction * self.params.body_mass * 10)

        # append current simulation data to internal array
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
                                            stomach_to_intestine_g_h,
                                            bac_percent))

    def simulate(self):
        """Simulates for the given total time in the parameters"""
        while self.time_hrs < self.params.total_time: # while still simulating
            self.step() # step the simulation
            self.time_hrs += self.sim_rate # increase the time passed

def main() -> None:
    """Main function"""
    st.set_page_config(page_title="BAC Simulation", layout="wide") # set tab title and layout style to wide

    st.title("Blood Alcohol Simulation") # create a title

    st.sidebar.header("Simulation Parameters") # create a header

    metab_values: dict[str, float] = {
        "Slow": 0.75, # Slow preset = 75% speed
        "Average": 1.0, # Average preset = 100% speed
        "Fast": 1.25, # Fast preset = 125% speed
    }

    aldh_values: dict[str, float] = {
        "Normal": 1.0, # Normal preset = 100% speed
        "Reduced": 0.7, # Reduced preset = 70% speed
        "Very reduced": 0.5, # Very reduced preset = 50% speed
    }
    # Note for both presets that the speeds don't translate directly to ethanol in/out but specific parts

    if "results" not in st.session_state: # if results value is not currently stored in the state
        st.session_state["results"] = None # initialize results in state to None

    with st.sidebar.form("simulation_form"): # put sliders in a sidebar form
        drinks: int = int(st.slider("Drinks consumed", 0, 12, 3, 1)) # 0-12, default 3, step 1
        duration: float = float(st.slider("Drinking duration (hours)", 0.1, 8.0, 2.0, 0.1, format="%.1f")) # 0.1-8.0, default 2.0, step 0.1
        food_eaten: int = int(st.slider("Food eaten (grams)", 0, 1500, 250, 50)) # 0-1500, default 250, step 50
        body_mass: int = int(st.slider("Body mass (kg)", 30, 150, 70, 1)) # 30-150, default 70, step 1
        body_water_fraction: int = int(st.slider("Body Water Fraction (%)", 50, 75, 68, 1)) # 50-75, default 68, step 1
        total_time: int = int(st.slider("Simulation time (hours)", 2, 32, 12, 1)) # 2-32, default 12, step 1

        metab_choice: str = str(st.selectbox("Ethanol metabolism preset", ["Slow", "Average", "Fast"], index=1)) # select box, default average
        aldh_choice: str = str(st.selectbox("ALDH efficiency", ["Normal", "Reduced", "Very reduced"], index=0)) # select box, default normal

        run: bool = bool(st.form_submit_button("Run simulation")) # run button bool

    if run: # if simulation has been run
        # assemble siulation parameters from sliders
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

        sim: Simulation = Simulation(params) # create new simulation with parameters
        sim.simulate() # simulate

        df: pd.DataFrame = pd.DataFrame([asdict(point) for point in sim.sim_data]) # easier to graph and do operations on every data point
        peak: SimulationData = max(sim.sim_data, key=lambda point: point.bac_percent) # max BAC
        final: SimulationData = sim.sim_data[-1] # final values

        drink_times: list[float] = [] # times in which alcohol was drunk
        if drinks == 1: # if there is only one drink
            drink_times = [0.0] # the drink time is at time 0
        elif drinks > 1: # if there is more than one drink
            drink_times = [i * duration / (drinks - 1) for i in range(drinks)] # drink times = i * drinks per hour for i in range(drinks)
        
        # store results dictionary in session state
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

    # return if there is no results in the session state (simulation has not run yet)
    if st.session_state["results"] is None:
        return

    results: dict[str, object] = cast(dict[str, object], st.session_state["results"]) # get results as dictionary

    df: pd.DataFrame = cast(pd.DataFrame, results["df"]) # pandas data frame for simulation data
    peak: SimulationData = cast(SimulationData, results["peak"]) # data for peak BAC
    final: SimulationData = cast(SimulationData, results["final"]) # final data
    drink_times: list[float] = cast(list[float], results["drink_times"]) # times in hours at which a drink was drunk
    displayed_total_time: int = cast(int, results["total_time"]) # total time of simulation
    displayed_params: Params = cast(Params, results["params"]) # params used to run the simulation
    displayed_metab_choice: str = cast(str, results["metab_choice"]) # metabolism choice
    displayed_aldh_choice: str = cast(str, results["aldh_choice"]) # aldh choice
    sim_rate: float = cast(float, results["sim_rate"]) # simulation rate

    # store current settings from the form
    current_settings: dict[str, object] = {
        "drinks": drinks,
        "duration": duration,
        "food_eaten": food_eaten,
        "body_mass": body_mass,
        "total_time": total_time,
        "metab_choice": metab_choice,
        "aldh_choice": aldh_choice,
    }

    # get the setting used to run the simulation currently being graphed
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

    # top main stats 
    col1, col2, col3, col4 = st.columns(4) # create 4 collums at the top
    col1.metric("Peak BAC", f"{peak.bac_percent:.3f}%") # write peak BAC
    col2.metric("Time to peak", f"{peak.time_hrs:.2f} h") # write time to peak BAC
    col3.metric("Time at 0.08%", threshold_interval(0.08)) # write window above legal limit
    col4.metric("BAC exposure", f"{auc_bac:.3f} %-h") # write BAC exposure (area under BAC curve)

    st.subheader("Graphs") # create graphs subheader

    graph_col1, graph_col2 = st.columns(2) # BAC and all compartments
    graph_col3, graph_col4 = st.columns(2) # absorption/elimination and acetaldehyde

    with graph_col1:
        st.write("BAC Over Time") # write graph header

        fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True) # create a graph that is 6x3.5 and a constrained layout
        ax.plot(df["time_hrs"], df["bac_percent"], label="BAC", linewidth=2) # plot the Bac over time

        add_drink_lines(ax) # add dashed line for each drink black and red for the last drink

        ax.set_xlabel("Time (hours)") # set x label of the graph
        ax.set_ylabel("BAC (%)") # set y label of the graph
        ax.set_title("BAC Over Time") # set graph title
        ax.set_xlim(0, displayed_total_time) # set the maximum x value to the displayed total time
        ax.grid(True) # turn on grid lines
        ax.legend(fontsize=7) # add a legend

        st.pyplot(fig, use_container_width=True) # put the graph on the website

    with graph_col2:
        st.write("Compartment Amounts") # write graph header

        fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True) # create a graph that is 6x3.5 and a constrained layout
        ax.plot(df["time_hrs"], df["stomach_ethanol_g"], label="Stomach ethanol", linewidth=2) # plot the stomach ethanol over time
        ax.plot(df["time_hrs"], df["intestine_ethanol_g"], label="Intestine ethanol", linewidth=2) # plot the intestine ethanol over time
        ax.plot(df["time_hrs"], df["body_ethanol_g"], label="Body ethanol", linewidth=2) # plot the body ethanol over time
        ax.plot(df["time_hrs"], df["acetaldehyde_g"], label="Acetaldehyde", linewidth=2) # plot the acetaldehyde over time
        ax.plot(df["time_hrs"], df["acetate_g"], label="Total acetate", linewidth=2) # plot the acetate over time

        add_drink_lines(ax) # add dashed line for each drink black and red for the last drink

        ax.set_xlabel("Time (hours)") # set x label of the graph
        ax.set_ylabel("Amount (g)") # set y label of the graph
        ax.set_title("Compartments") # set graph title
        ax.set_xlim(0, displayed_total_time) # set the maximum x value to the displayed total time
        ax.grid(True) # turn on grid lines
        ax.legend(fontsize=7) # add a legend

        st.pyplot(fig, use_container_width=True) # put the graph on the website

    with graph_col3:
        st.write("Absorption and Elimination Rates") # write graph header

        fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True) # create a graph that is 6x3.5 and a constrained layout

        ax.plot(df["time_hrs"], df["ethanol_absorption_g_h"], label="Ethanol absorption", linewidth=2) # graph ethanol absorption rate over time
        ax.plot(df["time_hrs"], df["ethanol_elimination_g_h"], label="Ethanol elimination", linewidth=2) # graph ethanol elimination rate over time
        ax.plot(df["time_hrs"], df["acetaldehyde_elimination_g_h"], label="Acetaldehyde elimination", linewidth=2) # graph acetaldehyde elimination over time

        add_drink_lines(ax) # add dashed line for each drink black and red for the last drink

        ax.set_xlabel("Time (hours)") # set x label of the graph
        ax.set_ylabel("Rate (g/h)") # set y label of the graph
        ax.set_title("Absorption and Elimination Rates") # set graph title
        ax.set_xlim(0, displayed_total_time) # set the maximum x value to the displayed total time
        ax.grid(True) # turn on grid lines
        ax.legend(fontsize=7) # add a legend

        st.pyplot(fig, use_container_width=True) # put the graph on the website

    with graph_col4:
        st.write("Acetaldehyde Over Time") # write graph header

        fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True) # create a graph that is 6x3.5 and a constrained layout

        ax.plot(df["time_hrs"], df["acetaldehyde_g"], label="Acetaldehyde amount", linewidth=2)

        add_drink_lines(ax) # add dashed line for each drink black and red for the last drink

        ax.set_xlabel("Time (hours)") # set x label of the graph
        ax.set_ylabel("Acetaldehyde (g)") # set y label of the graph
        ax.set_title("Acetaldehyde Over Time") # set graph title
        ax.set_xlim(0, displayed_total_time) # set the maximum x value to the displayed total time
        ax.grid(True) # turn on grid lines
        ax.legend(fontsize=7) # add a legend

        st.pyplot(fig, use_container_width=True) # put the graph on the website

    st.subheader("Extra Stats") # write header

    total_absorbed_g: float = float((df["ethanol_absorption_g_h"] * sim_rate).sum()) # calculate the total absorbed ethanol
    total_metabolized_g: float = float((df["ethanol_elimination_g_h"] * sim_rate).sum()) # calculate the total metabolized ethanol
    max_absorption_rate: float = float(df["ethanol_absorption_g_h"].max()) # calculate the max absorption rate
    max_elimination_rate: float = float(df["ethanol_elimination_g_h"].max()) # calculate the max elimination rate

    peak_acetaldehyde = df.loc[df["acetaldehyde_g"].idxmax()] # get time of peak acetaldehyde
    # calculate peak acetaldehyde and the time at peak
    peak_acetaldehyde_g: float = float(peak_acetaldehyde["acetaldehyde_g"]) # type: ignore
    peak_acetaldehyde_time: float = float(peak_acetaldehyde["time_hrs"]) # type: ignore

    acetaldehyde_exposure: float = float((df["acetaldehyde_concentration_mmol_l"] * sim_rate).sum()) # calculate the area under the acetaldehyde graph

    alcohol_fully_absorbed_text: str = time_when_alcohol_fully_absorbed() # get the time at which alcohol is fully absorbed
    alcohol_fully_metabolized_text: str = time_when_alcohol_fully_metabolized() # get the time at which alcohol is fully metabolized

    # BAC timing stats table
    bac_timing_stats: pd.DataFrame = pd.DataFrame([
        {"Stat": "Time at 0.08% BAC", "Value": threshold_interval(0.08)},
        {"Stat": "Time at 0.05% BAC", "Value": threshold_interval(0.05)},
        {"Stat": "Time at 0.02% BAC", "Value": threshold_interval(0.02)},
        {"Stat": "BAC exposure", "Value": f"{auc_bac:.3f} %-h"},
    ])

    # ethanol stats table
    ethanol_stats: pd.DataFrame = pd.DataFrame([
        {"Stat": "Total ethanol absorbed", "Value": f"{total_absorbed_g:.2f} g"},
        {"Stat": "Total ethanol metabolized", "Value": f"{total_metabolized_g:.2f} g"},
        {"Stat": "Alcohol fully absorbed", "Value": alcohol_fully_absorbed_text},
        {"Stat": "Alcohol fully metabolized", "Value": alcohol_fully_metabolized_text},
        {"Stat": "Max absorption rate", "Value": f"{max_absorption_rate:.2f} g/h"},
        {"Stat": "Max elimination rate", "Value": f"{max_elimination_rate:.2f} g/h"},
    ])

    # metabolism stats table
    metabolism_stats: pd.DataFrame = pd.DataFrame([
        {"Stat": "Peak acetaldehyde", "Value": f"{peak_acetaldehyde_g:.4f} g at {peak_acetaldehyde_time:.2f} h"},
        {"Stat": "Acetaldehyde exposure", "Value": f"{acetaldehyde_exposure:.4f} mmol/L·h"},
        {"Stat": "Final body ethanol", "Value": f"{final.body_ethanol_g:.2f} g"},
        {"Stat": "Final acetate", "Value": f"{final.acetate_g:.2f} g"},
    ])

    stat_col1, stat_col2, stat_col3 = st.columns(3) # BAC Timing, Ethanol Movement, Metabolism

    with stat_col1:
        st.markdown("#### BAC Timing") # write header 4
        st.dataframe(bac_timing_stats, hide_index=True, use_container_width=True) # BAC timing stats table

    with stat_col2:
        st.markdown("#### Ethanol Movement") # write header 4
        st.dataframe(ethanol_stats, hide_index=True, use_container_width=True) # ethanol stats table

    with stat_col3:
        st.markdown("#### Metabolism") # write header 4
        st.dataframe(metabolism_stats, hide_index=True, use_container_width=True) # metabolism stats table

    with st.expander("Show raw data"): # expander to see full data
        st.dataframe(df) # all simulation data as a table

if __name__ == "__main__":
    main()
