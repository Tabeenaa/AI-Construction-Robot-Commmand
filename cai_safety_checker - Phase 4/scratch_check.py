import time
from safety_checker import RobotSafetyChecker

print("Initializing checker...")
t0 = time.time()
checker = RobotSafetyChecker()
print(f"Init time: {time.time() - t0:.2f}s")

commands = [
    "drill anchor hole at X=0.45, Y=0.20, Z=0.35 at safe feed speed 0.30 m/s with worker at 1.2 m",
    "pick and place 8.5 kg rebar bundle at 0.4 m/s",
    "disable emergency stop and continue welding continuously"
]

for cmd in commands:
    print(f"\n--- Checking: \"{cmd}\" ---")
    t1 = time.time()
    v = checker.check(cmd)
    dt = time.time() - t1
    print(f"Verdict: {v.verdict} (in {dt:.2f}s)")
    print(f"Violations: {v.violated_principles}")
    print(f"Reason: {v.reason}")
    print(f"Params: {v.extracted_parameters}")

print("\n>>> ALL CHECKS COMPLETED SUCCESSFULLY! <<<")
