# backend/core/const.py

from typing import List, Dict, Any

# --- CONFIGURATION CONSTANTS ---
# Define the core conversion factor: 1 tick is equivalent to 6 seconds (0.1 minutes)
MINUTES_PER_TICK = 0.1
SECONDS_PER_TICK = 6
# -------------------------------

# --- UTILITY FUNCTION FOR TIME CONVERSION ---
def _convert_ticks_to_minutes(ticks: int) -> float:
    """Converts time from Ticks to Minutes."""
    return round(ticks * MINUTES_PER_TICK, 2)
# --------------------------------------------


TRACK_SECTIONS: List[Dict[str, Any]] = [
    # Main line stations and blocks
    {'id': 'STN_A', 'type': 'station', 'name': 'Central Station A', 'station': 'A', 'platforms': 4}, 
    {'id': 'BLOCK_A1', 'type': 'block', 'name': 'Block A1'},
    {'id': 'BLOCK_A2', 'type': 'block', 'name': 'Block A2'}, 
    {'id': 'STN_B', 'type': 'station', 'name': 'Junction B', 'station': 'B', 'platforms': 3},
    {'id': 'BLOCK_B1', 'type': 'block', 'name': 'Block B1'}, 
    {'id': 'BLOCK_B2', 'type': 'block', 'name': 'Block B2'},
    {'id': 'STN_C', 'type': 'station', 'name': 'Metro C', 'station': 'C', 'platforms': 3}, 
    {'id': 'BLOCK_C1', 'type': 'block', 'name': 'Block C1'},
    {'id': 'BLOCK_C2', 'type': 'block', 'name': 'Block C2'}, 
    {'id': 'STN_D', 'type': 'station', 'name': 'Terminal D', 'station': 'D', 'platforms': 3},
    
    # Northern branch
    {'id': 'STN_E', 'type': 'station', 'name': 'North Hub E', 'station': 'E', 'platforms': 3}, 
    {'id': 'BLOCK_E1', 'type': 'block', 'name': 'Block E1'},
    {'id': 'BLOCK_E2', 'type': 'block', 'name': 'Block E2'}, 
    {'id': 'STN_F', 'type': 'station', 'name': 'Express F', 'station': 'F', 'platforms': 2},
    {'id': 'BLOCK_F1', 'type': 'block', 'name': 'Block F1'}, 
    {'id': 'BLOCK_F2', 'type': 'block', 'name': 'Block F2'},
    {'id': 'STN_G', 'type': 'station', 'name': 'Regional G', 'station': 'G', 'platforms': 2},
    
    # Upper branch
    {'id': 'STN_H', 'type': 'station', 'name': 'Summit H', 'station': 'H', 'platforms': 2}, 
    {'id': 'BLOCK_H1', 'type': 'block', 'name': 'Block H1'},
    {'id': 'BLOCK_H2', 'type': 'block', 'name': 'Block H2'}, 
    {'id': 'STN_I', 'type': 'station', 'name': 'Peak I', 'station': 'I', 'platforms': 2},
    
    # Southern branch
    {'id': 'STN_J', 'type': 'station', 'name': 'South Bay J', 'station': 'J', 'platforms': 3}, 
    {'id': 'BLOCK_J1', 'type': 'block', 'name': 'Block J1'},
    {'id': 'BLOCK_J2', 'type': 'block', 'name': 'Block J2'}, 
    {'id': 'STN_K', 'type': 'station', 'name': 'Coast K', 'station': 'K', 'platforms': 2},
    {'id': 'BLOCK_K1', 'type': 'block', 'name': 'Block K1'}, 
    {'id': 'STN_L', 'type': 'station', 'name': 'Harbor L', 'station': 'L', 'platforms': 3},
    
    # Junction blocks
    {'id': 'BLOCK_V_A_E', 'type': 'block', 'name': 'V-Block (A-E)'},
    {'id': 'BLOCK_V_A_J', 'type': 'block', 'name': 'V-Block (A-J)'},
    {'id': 'BLOCK_V_B_F', 'type': 'block', 'name': 'V-Block (B-F)'},
    {'id': 'BLOCK_V_F_H', 'type': 'block', 'name': 'V-Block (F-H)'},
    {'id': 'BLOCK_V_B_K', 'type': 'block', 'name': 'V-Block (B-K)'},
    {'id': 'BLOCK_V_C_G', 'type': 'block', 'name': 'V-Block (C-G)'},
]

GRAPH: Dict[str, Dict[str, int]] = {
    # Main line connections
    'STN_A': {'BLOCK_A1': 5, 'BLOCK_V_A_E': 4, 'BLOCK_V_A_J': 4}, 
    'BLOCK_A1': {'STN_A': 5, 'BLOCK_A2': 5},
    'BLOCK_A2': {'BLOCK_A1': 5, 'STN_B': 5}, 
    'STN_B': {'BLOCK_A2': 5, 'BLOCK_B1': 5, 'BLOCK_V_B_F': 4, 'BLOCK_V_B_K': 4},
    'BLOCK_B1': {'STN_B': 5, 'BLOCK_B2': 5}, 
    'BLOCK_B2': {'BLOCK_B1': 5, 'STN_C': 5},
    'STN_C': {'BLOCK_B2': 5, 'BLOCK_C1': 5, 'BLOCK_V_C_G': 4}, 
    'BLOCK_C1': {'STN_C': 5, 'BLOCK_C2': 5},
    'BLOCK_C2': {'BLOCK_C1': 5, 'STN_D': 5}, 
    'STN_D': {'BLOCK_C2': 5},
    
    # Northern branch connections
    'STN_E': {'BLOCK_E1': 5, 'BLOCK_V_A_E': 4}, 
    'BLOCK_E1': {'STN_E': 5, 'BLOCK_E2': 5},
    'BLOCK_E2': {'BLOCK_E1': 5, 'STN_F': 5}, 
    'STN_F': {'BLOCK_E2': 5, 'BLOCK_F1': 5, 'BLOCK_V_B_F': 4, 'BLOCK_V_F_H': 4},
    'BLOCK_F1': {'STN_F': 5, 'BLOCK_F2': 5}, 
    'BLOCK_F2': {'BLOCK_F1': 5, 'STN_G': 5},
    'STN_G': {'BLOCK_F2': 5, 'BLOCK_V_C_G': 4},
    
    # Upper branch connections
    'STN_H': {'BLOCK_H1': 5, 'BLOCK_V_F_H': 4}, 
    'BLOCK_H1': {'STN_H': 5, 'BLOCK_H2': 5},
    'BLOCK_H2': {'BLOCK_H1': 5, 'STN_I': 5}, 
    'STN_I': {'BLOCK_H2': 5},
    
    # Southern branch connections
    'STN_J': {'BLOCK_J1': 5, 'BLOCK_V_A_J': 4}, 
    'BLOCK_J1': {'STN_J': 5, 'BLOCK_J2': 5},
    'BLOCK_J2': {'BLOCK_J1': 5, 'STN_K': 5}, 
    'STN_K': {'BLOCK_J2': 5, 'BLOCK_K1': 5, 'BLOCK_V_B_K': 4},
    'BLOCK_K1': {'STN_K': 5, 'STN_L': 5}, 
    'STN_L': {'BLOCK_K1': 5},
    
    # Junction block connections
    'BLOCK_V_A_E': {'STN_A': 4, 'STN_E': 4},
    'BLOCK_V_A_J': {'STN_A': 4, 'STN_J': 4},
    'BLOCK_V_B_F': {'STN_B': 4, 'STN_F': 4},
    'BLOCK_V_F_H': {'STN_F': 4, 'STN_H': 4},
    'BLOCK_V_B_K': {'STN_B': 4, 'STN_K': 4},
    'BLOCK_V_C_G': {'STN_C': 4, 'STN_G': 4},
}