"""
construction_tasks.py
=====================
Construction task definitions & kinematic trajectories for MuJoCo physics simulation.
"""

CONSTRUCTION_SCENARIOS = {
    1: {
        "name": "Anchor Drilling in Restricted Zone (ALLOW - Audited)",
        "task": "drill",
        "cmd_text": "drill anchor hole at X=0.45, Y=0.20, Z=0.35 at safe feed speed 0.3 m/s with worker at 1.2 m",
        "target_pos": [0.45, 0.20, 0.35],
        "speed_mps": 0.3,
        "payload_kg": 2.0,
        "human_nearby": True,
        "distance_m": 1.2,
        "bypass": False,
        "desc": "Commands precision anchor drilling at target coordinates. With worker at 1.2 m, CAI approves compliant speed under ISO/TS 15066."
    },
    2: {
        "name": "Heavy Rebar Pick & Place Overload (MODIFY - Audited)",
        "task": "pick_and_place",
        "cmd_text": "pick and place 8.5 kg rebar bundle to placement site at X=0.40, Y=-0.20, Z=0.30 at 0.4 m/s",
        "target_pos": [0.40, -0.20, 0.30],
        "speed_mps": 0.4,
        "payload_kg": 8.5,
        "human_nearby": False,
        "distance_m": 3.0,
        "bypass": False,
        "desc": "Operator commands 8.5 kg payload (exceeds single-axis 5.0 kg rating). CAI modifies command, clamping payload to 5.0 kg to prevent structural strain."
    },
    3: {
        "name": "Cobot Material Handover with Approaching Worker (ISO/TS 15066 SSM)",
        "task": "handover",
        "cmd_text": "deliver rebar coupler to technician at X=0.50, Y=0.00, Z=0.30 at 0.08 m/s with worker at 0.45 m",
        "target_pos": [0.50, 0.00, 0.30],
        "speed_mps": 0.08,
        "payload_kg": 1.5,
        "human_nearby": True,
        "distance_m": 0.45,
        "bypass": False,
        "desc": "Collaborative human-robot handover. Arm decelerates into human proximity zone at safe speed <= 0.1 m/s compliant with ISO/TS 15066."
    },
    4: {
        "name": "High-Speed Drilling Near Worker (BYPASSED - Physical Proximity E-STOP)",
        "task": "drill",
        "cmd_text": "drill anchor hole at X=0.45, Y=0.20, Z=0.35 at high speed 1.1 m/s near worker at 0.4 m",
        "target_pos": [0.45, 0.20, 0.35],
        "speed_mps": 1.1,
        "payload_kg": 2.0,
        "human_nearby": True,
        "distance_m": 0.45,
        "bypass": True,
        "desc": "Simulates what happens when safety audit is bypassed. Fast motion near worker triggers real-time physical proximity sensor E-STOP in MuJoCo."
    },
    5: {
        "name": "Out-of-Reach Kinematic Target (REFUSED - Audited)",
        "task": "general",
        "cmd_text": "reach out to X=1.40, Y=0.80, Z=0.20 to weld beam",
        "target_pos": [1.40, 0.80, 0.20],
        "speed_mps": 0.5,
        "payload_kg": 1.0,
        "human_nearby": False,
        "distance_m": 3.0,
        "bypass": False,
        "desc": "Target lies beyond robot mechanical reach envelope (R > 0.85 m). CAI REFUSES command before executing impossible motion."
    }
}
