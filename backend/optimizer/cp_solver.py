from ortools.sat.python import cp_model
from typing import Dict, Any, List
import json

# --- Configuration (Tuned for your existing network) ---
# Critical intersection blocks (where trains conflict)
CRITICAL_BLOCKS = ["BLOCK_A2", "BLOCK_V_D2_A2", "BLOCK_D2", "STN_B"] 
TIME_SLOT_SECONDS = 30 # Discretization step
HORIZON_MINUTES = 10 
HORIZON_TICKS = int((HORIZON_MINUTES * 60) / TIME_SLOT_SECONDS) # 20 ticks

# Simple mapping of train name to assumed time to clear a block (in ticks)
TRANSIT_TICKS_MAP = {
    "Express": 4, # 2.0 min
    "Sprinter": 3, # 1.5 min
    "Commuter": 5, # 2.5 min
    "Cargo": 6, # 3.0 min (Longest)
}

def get_transit_ticks(train_name: str) -> int:
    """Helper to get transit time based on train type."""
    return TRANSIT_TICKS_MAP.get(train_name, 5)

def solve_junction_precedence(
    current_state: Dict[str, Any], 
    train_static_data: Dict[str, Any],
    sim_time_seconds: int
) -> Dict[str, Any]:
    """
    Finds the optimal, conflict-free sequence for trains approaching the critical junction.
    """
    model = cp_model.CpModel()
    
    # --- 1. Filter Relevant Trains ---
    approaching_trains = []
    
    for train in current_state['trains']:
        # Check if the train is currently in or waiting for a critical block (based on its route)
        current_idx = train_static_data[train['id']]['currentRouteIndex']
        route = train_static_data[train['id']]['route']
        
        # Check the next 2 blocks in the route
        if current_idx < len(route) - 1 and (route[current_idx] in CRITICAL_BLOCKS or route[current_idx + 1] in CRITICAL_BLOCKS):
            # Augment train data with static info for the solver
            train_info = {
                'id': train['id'],
                'name': train['name'],
                'priority': train_static_data[train['id']]['priority'],
                'current_section': train['section'],
                'next_block': route[current_idx + 1] if current_idx < len(route) - 1 else None,
                'transit_ticks': get_transit_ticks(train['name'])
            }
            approaching_trains.append(train_info)

    if len(approaching_trains) < 2:
        return {"recommendation": "NO_CONFLICT", "decisions": {}}

    print(f"OR-Solver running for {len(approaching_trains)} critical trains.")
    
    # --- 2. Define Variables ---
    train_intervals = {}
    
    for train in approaching_trains:
        # Create an interval for the train's occupation of the *first* critical block it is approaching
        block_to_enter = train['next_block'] if train['next_block'] in CRITICAL_BLOCKS else train['current_section']
        if block_to_enter in CRITICAL_BLOCKS:
            duration = train['transit_ticks']

            # Start time must be within the horizon
            start_var = model.NewIntVar(0, HORIZON_TICKS, f'start_{train["id"]}_{block_to_enter}')
            end_var = model.NewIntVar(0, HORIZON_TICKS + duration, f'end_{train["id"]}_{block_to_enter}')
            
            interval = model.NewIntervalVar(start_var, duration, end_var, f'interval_{train["id"]}_{block_to_enter}')
            train_intervals[(train['id'], block_to_enter)] = interval

    # --- 3. Define Constraints ---

    # Constraint: No two trains can occupy the same critical block simultaneously
    for block_id in CRITICAL_BLOCKS:
        intervals_on_block = [
            train_intervals[(t['id'], block_id)] 
            for t in approaching_trains 
            if (t['id'], block_id) in train_intervals
        ]
        if len(intervals_on_block) > 1:
            model.AddNoOverlap(intervals_on_block)

    # --- 4. Objective Function ---
    
    # Objective: Minimize the sum of weighted entry times (weighted by priority).
    total_weighted_start_time = []
    
    for train in approaching_trains:
        # Lower priority number (e.g., 10) means higher importance (weight).
        weight = 100 - train['priority']
        
        block_to_enter = train['next_block'] if train['next_block'] in CRITICAL_BLOCKS else train['current_section']

        if (train['id'], block_to_enter) in train_intervals:
            start_expr = train_intervals[(train['id'], block_to_enter)].StartExpr()
            # If the train is already waiting, we penalize the wait time relative to now (0)
            total_weighted_start_time.append(weight * start_expr) 

    if total_weighted_start_time:
        model.Minimize(sum(total_weighted_start_time))
    else:
        return {"recommendation": "NO_CONFLICT", "decisions": {}}


    # --- 5. Solve and Extract Results ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 0.5 # Fast re-optimization is critical

    status = solver.Solve(model)
    
    decisions = {}
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for train in approaching_trains:
            block_to_enter = train['next_block'] if train['next_block'] in CRITICAL_BLOCKS else train['current_section']
            
            if (train['id'], block_to_enter) in train_intervals:
                start_tick = solver.Value(train_intervals[(train['id'], block_to_enter)].StartExpr())
                start_time_seconds = start_tick * TIME_SLOT_SECONDS

                decisions[train['id']] = {
                    "action": "PROCEED" if start_tick == 0 else "HOLD",
                    "start_sim_time_seconds": sim_time_seconds + start_time_seconds,
                    "wait_time_seconds": start_time_seconds,
                    "target_block": block_to_enter,
                    "reason": f"Optimally scheduled to enter {block_to_enter} at T+{start_time_seconds}s to minimize overall delay."
                }
        
        return {"recommendation": "OPTIMAL_SCHEDULE_FOUND", "decisions": decisions}
    
    else:
        print(f"Solver failed to find a solution: {solver.StatusName(status)}")
        return {"recommendation": "NO_SOLUTION_FOUND", "decisions": {}}
