"""
construction_tasks.py
=====================
Construction task definitions & kinematic trajectories for MuJoCo physics simulation.

Home EE position (fully-upright): [0.0, 0.0, 1.306]
All targets chosen for large, clearly-observable motion from home.
"""

CONSTRUCTION_SCENARIOS = {
    1: {
        "name": "Anchor Drilling in Restricted Zone (ALLOW - Audited)",
        "task": "drill",
        # From home [0,0,1.3] → arm extends forward-low to drill position (~80cm travel)
        "cmd_text": "drill anchor hole at X=0.55, Y=0.10, Z=0.40 at safe feed speed 0.3 m/s with worker at 1.2 m",
        "target_pos": [0.55, 0.10, 0.40],
        "speed_mps": 0.3,
        "payload_kg": 2.0,
        "human_nearby": True,
        "distance_m": 1.2,
        "bypass": False,
        "desc": "Commands precision anchor drilling at target coordinates. With worker at 1.2m, CAI approves compliant speed under ISO/TS 15066."
    },
    2: {
        "name": "Heavy Rebar Pick & Place Overload (MODIFY - Audited)",
        "task": "pick_and_place",
        # Arm swings left and low (~85cm travel)
        "cmd_text": "pick and place 8.5 kg rebar bundle to placement site at X=0.30, Y=0.55, Z=0.35 at 0.4 m/s",
        "target_pos": [0.30, 0.55, 0.35],
        "speed_mps": 0.4,
        "payload_kg": 8.5,
        "human_nearby": False,
        "distance_m": 3.0,
        "bypass": False,
        "desc": "Operator commands 8.5 kg payload (exceeds single-axis 5.0 kg rating). CAI modifies command, clamping payload to 5.0 kg."
    },
    3: {
        "name": "Cobot Material Handover with Approaching Worker (ISO/TS 15066 SSM)",
        "task": "handover",
        # Arm extends forward at mid-height for handover (~75cm travel)
        "cmd_text": "deliver rebar coupler to technician at X=0.60, Y=0.00, Z=0.55 at 0.08 m/s with worker at 0.45 m",
        "target_pos": [0.60, 0.00, 0.55],
        "speed_mps": 0.08,
        "payload_kg": 1.5,
        "human_nearby": True,
        "distance_m": 0.45,
        "bypass": False,
        "desc": "Collaborative human-robot handover. Arm decelerates into human proximity zone at safe speed <= 0.1 m/s (ISO/TS 15066)."
    },
    4: {
        "name": "High-Speed Drilling Near Worker (BYPASSED - Physical Proximity E-STOP)",
        "task": "drill",
        # Same target as scenario 1 but bypassed safety — E-STOP will trigger
        "cmd_text": "drill anchor hole at X=0.55, Y=0.10, Z=0.40 at high speed 1.1 m/s near worker at 0.4 m",
        "target_pos": [0.55, 0.10, 0.40],
        "speed_mps": 1.1,
        "payload_kg": 2.0,
        "human_nearby": True,
        "distance_m": 0.40,
        "bypass": True,
        "desc": "Simulates what happens when safety audit is bypassed. Fast motion near worker triggers real-time physical proximity E-STOP."
    },
    5: {
        "name": "Out-of-Reach Kinematic Target (REFUSED - Audited)",
        "task": "general",
        # Target far outside KUKA reach envelope (R > 0.85m) → CAI REFUSES
        "cmd_text": "reach out to X=1.50, Y=1.00, Z=0.10 to weld beam",
        "target_pos": [1.50, 1.00, 0.10],
        "speed_mps": 0.5,
        "payload_kg": 1.0,
        "human_nearby": False,
        "distance_m": 3.0,
        "bypass": False,
        "desc": "Target lies beyond robot mechanical reach envelope. CAI REFUSES command before executing impossible motion."
    }
}
