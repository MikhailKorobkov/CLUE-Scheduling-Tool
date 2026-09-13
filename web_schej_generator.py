import streamlit as st
import pandas as pd
import csv
import io
from ortools.sat.python import cp_model

# --- CONFIGURATION ---
NUMBER_TO_DAY_NAME = {
    0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 
    4: "Thursday", 5: "Friday", 6: "Saturday"
}

DAY_NAME_TO_NUMBER = {v.lower(): k for k, v in NUMBER_TO_DAY_NAME.items()}
DAY_NAME_TO_NUMBER.update({"sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6})

def parse_days_input(day_string):
    parsed_days = []
    if not day_string or not str(day_string).strip(): return []
    parts = [d.strip().replace('"', '') for d in str(day_string).split(',') if d.strip()]
    for part in parts:
        if part.lower() in DAY_NAME_TO_NUMBER:
            parsed_days.append(DAY_NAME_TO_NUMBER[part.lower()])
        else:
            try: parsed_days.append(int(part) - 1)
            except: pass
    return sorted(list(set(parsed_days)))

def load_worker_data(uploaded_file):
    workers = []
    w_id = 0
    content = uploaded_file.getvalue().decode("utf-8-sig")
    reader = csv.reader(io.StringIO(content))
    
    try: next(reader)
    except: pass
    
    for row in reader:
        if not row or not row[0].strip(): continue
        r = row + [''] * (5 - len(row))
        try:
            def safe_int(val, default):
                return int(val.strip()) if val and str(val).strip() else default

            w = {
                'id': w_id,
                'name': r[0].strip(),
                'pref': parse_days_input(r[1]),
                'avail': parse_days_input(r[2]),
                'min_s': safe_int(r[3], 0),
                'max_s': safe_int(r[4], 99)
            }
            w['all_valid'] = sorted(list(set(w['pref'] + w['avail'])))
            workers.append(w)
            w_id += 1
        except: continue
    return workers

def create_base_model(workers, num_days, req_staff):
    model = cp_model.CpModel()
    sh = {}
    objective_terms = []
    
    COST_NON_PREF = 1
    COST_UNAVAIL = 1000

    for w in workers:
        for d in range(num_days):
            shift = model.NewBoolVar(f"w{w['id']}_d{d}")
            sh[(w['id'], d)] = shift

            if d in w['pref']: pass
            elif d in w['avail']: objective_terms.append(shift * COST_NON_PREF)
            else: objective_terms.append(shift * COST_UNAVAIL)

    for d in range(num_days):
        model.Add(sum(sh[(w['id'], d)] for w in workers) == req_staff[d])

    for w in workers:
        all_shifts = [sh[(w['id'], d)] for d in range(num_days)]
        model.Add(sum(all_shifts) >= w['min_s'])
        model.Add(sum(all_shifts) <= w['max_s'])

    return model, sh, objective_terms

class SolutionCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, sh_dict, limit):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__sh_dict = sh_dict
        self.__limit = limit
        self.solutions = []

    def on_solution_callback(self):
        current_solution = {}
        for key, var in self.__sh_dict.items():
            current_solution[key] = self.Value(var)
        self.solutions.append(current_solution)
        
        if len(self.solutions) >= self.__limit:
            self.StopSearch()

def solve_all_optimal_schedules(workers, num_days, req_staff, max_solutions):
    # Pass 1: Find best score
    model1, sh1, obj_terms1 = create_base_model(workers, num_days, req_staff)
    model1.Minimize(sum(obj_terms1))
    
    solver1 = cp_model.CpSolver()
    status1 = solver1.Solve(model1)
    
    if status1 not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return None, status1, []
        
    best_score = int(solver1.ObjectiveValue())
    
    # Pass 2: Find all solutions tying the best score
    model2, sh2, obj_terms2 = create_base_model(workers, num_days, req_staff)
    model2.Add(sum(obj_terms2) == best_score)
    
    solver2 = cp_model.CpSolver()
    solver2.parameters.enumerate_all_solutions = True
    
    collector = SolutionCollector(sh2, limit=max_solutions)
    status2 = solver2.Solve(model2, collector)
    
    return best_score, status1, collector.solutions

def generate_schedule_dataframe(solution_dict, workers, num_days):
    rows = []
    max_slots = 0
    
    for d in range(num_days):
        day_name = NUMBER_TO_DAY_NAME.get(d, f"Day {d+1}")
        scheduled_staff = sorted([w['name'] for w in workers if solution_dict.get((w['id'], d)) == 1])
        rows.append([day_name] + scheduled_staff)
        if len(scheduled_staff) > max_slots: 
            max_slots = len(scheduled_staff)
    
    headers = ["Day"] + [f"Staff {i+1}" for i in range(max_slots)]
    
    for row in rows:
        while len(row) < len(headers): row.append("")
        
    return pd.DataFrame(rows, columns=headers)

# --- STREAMLIT UI ---
st.set_page_config(page_title="Automated Shift Scheduler", layout="wide")

st.title("📅 Automated Shift Scheduler")
st.markdown("Upload your team's availability, set your daily needs, and generate optimal schedules.")

with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload Worker CSV", type=["csv"])
    
    st.header("2. Schedule Scope")
    num_days = st.number_input("How many days to schedule?", min_value=1, max_value=31, value=5)
    
    max_options = st.slider("Max Schedule Options to Generate", min_value=1, max_value=20, value=5, 
                            help="Find alternative schedules that have the exact same optimality score.")
    
    st.header("3. Daily Requirements")
    req_staff = []
    for i in range(num_days):
        day_label = NUMBER_TO_DAY_NAME.get(i, f"Day {i+1}")
        
        # Sets Monday and Wednesday to 3 by default, and others to 2
        default_staff = 3 if day_label in ["Monday", "Wednesday"] else 2
        
        val = st.number_input(
            f"{day_label} Staff Needed:", 
            min_value=0, 
            max_value=50, 
            value=default_staff, 
            key=f"req_{i}"
        )
        req_staff.append(val)
        
    run_solver = st.button("Generate Schedules", type="primary", use_container_width=True)

if run_solver:
    if not uploaded_file:
        st.error("Please upload a worker CSV file first!")
    else:
        workers = load_worker_data(uploaded_file)
        if not workers:
            st.error("No valid data found in CSV.")
        else:
            with st.spinner("Calculating optimal schedules..."):
                best_score, status, solutions = solve_all_optimal_schedules(workers, num_days, req_staff, max_options)
            
            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] and solutions:
                st.success(f"✅ Found {len(solutions)} schedule option(s) matching the best possible score!")
                
                forced = best_score // 1000
                non_pref = best_score % 1000
                total_shifts = sum(req_staff)
                perfect_shifts = total_shifts - forced - non_pref

                col1, col2, col3 = st.columns(3)
                col1.metric("Perfect Shifts (Preferred)", perfect_shifts)
                col2.metric("Available Shifts Used", non_pref)
                col3.metric("Unavailable Shifts Forced", forced, delta_color="inverse")
                
                if forced > 0:
                    st.warning("⚠️ The solver had to break availability to satisfy the math (e.g., someone is scheduled on a day they are unavailable).")
                elif non_pref > 0:
                    st.info("The schedule is balanced using some non-preferred 'Available' days.")
                else:
                    st.balloons()
                    st.success("Perfect Schedule! Everyone is working only on their preferred days.")

                st.markdown("### Schedule Options")
                st.markdown("All tabs below represent equal-ranked solutions. Choose the one you like best!")
                
                tabs = st.tabs([f"Option {i+1}" for i in range(len(solutions))])
                
                for i, tab in enumerate(tabs):
                    with tab:
                        df = generate_schedule_dataframe(solutions[i], workers, num_days)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"📥 Download Option {i+1} as CSV",
                            data=csv_data,
                            file_name=f"optimized_schedule_option_{i+1}.csv",
                            mime="text/csv",
                            key=f"dl_btn_{i}"
                        )
            else:
                st.error("❌ Impossible to generate schedule.")
                st.markdown("""
                **Why did this happen?**
                * Check for **Oversupply**: Total Min Shifts required by workers > Total Staff Demand.
                * Check for **Undersupply**: Total Max Shifts allowed by workers < Total Staff Demand.
                """)