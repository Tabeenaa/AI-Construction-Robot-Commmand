# SolidWorks Custom Robot Arm Integration Guide
### From `Assem1.SLDASM` to MuJoCo & Constitutional AI Autonomous Execution
**Designed for Member 3 (Perception & CAD / Digital Twin Lead)**

---

## 1. Overview of Your CAD Files in `d:\robot\`

Your workspace already contains a complete mechanical design of a custom robotic arm:

```
d:\robot\
├── Assem1.SLDASM            # Master SolidWorks Assembly
├── base_model.SLDPRT        # Link 0: Stationary Mount / Base Pedestal
├── rotor.SLDPRT             # Link 1: Base Yaw Turret
├── upper_arm_link.SLDPRT    # Link 2: Primary Shoulder Pitch Link
├── wrist model.SLDPRT       # Link 3 & 4: Wrist Articulation
├── driver.SLDPRT            # Link 5: Actuator Drive Link
├── gripper_actuator.SLDPRT  # Link 6: End-Effector Adapter
└── hand gripper.SLDPRT      # Link 7: Construction Hand Gripper
```

---

## 2. Recommended Kinematic Hierarchy

When exporting to **URDF** (Unified Robot Description Format), follow this parent-child kinematic tree:

```mermaid
graph TD
    Base[base_model.SLDPRT - Ground/Pedestal] -->|Joint 1: Revolute Yaw (Z)| Rotor[rotor.SLDPRT]
    Rotor -->|Joint 2: Revolute Pitch (Y)| UpperArm[upper_arm_link.SLDPRT]
    UpperArm -->|Joint 3: Revolute Pitch (Y)| Wrist[wrist model.SLDPRT]
    Wrist -->|Joint 4: Revolute Roll/Pitch| Driver[driver.SLDPRT]
    Driver -->|Joint 5: Revolute Wrist Twist| GripperActuator[gripper_actuator.SLDPRT]
    GripperActuator -->|Joint 6: Prismatic/Revolute| HandGripper[hand gripper.SLDPRT]
    HandGripper -->|Site| AttachmentSite["attachment_site (End-Effector TCP)"]
```

---

## 3. Step-by-Step Export Process (SolidWorks to URDF)

### Step 1: Install the SolidWorks to URDF Plugin
1. Download the official **ROS SolidWorks to URDF Exporter** (`sw2urdf`) from:
   `http://wiki.ros.org/sw_urdf_exporter`
2. Install the `.exe` and restart SolidWorks.

### Step 2: Configure Coordinate Reference Frames in SolidWorks
1. In SolidWorks, open `d:\robot\Assem1.SLDASM`.
2. Go to **Features** $\to$ **Reference Geometry** $\to$ **Coordinate System**.
3. Create a coordinate system at each joint axis:
   - Make sure the **Z-axis** points along the rotational axis of each motor.
   - Place the origin at the center of the bearing/joint shaft.

### Step 3: Launch the Exporter
1. Click **Tools** $\to$ **Export to URDF**.
2. Define the links according to the hierarchy shown above.
3. For each link:
   - Select the corresponding `.SLDPRT` component.
   - Choose the joint type (e.g. `continuous` or `revolute`).
   - Set joint safety limits in radians (e.g., Joint 1: `[-3.14, 3.14]`, Joint 2: `[-2.0, 2.0]`).
4. Click **Preview and Export**.
5. Save the output package as `custom_arm_urdf/` into:
   `d:\robot\cai_safety_checker - Phase 4\models\custom_arm\`

---

## 4. Converting URDF to MuJoCo MJCF Format

MuJoCo natively loads URDF files or compiles them into optimized MJCF XML.

1. **Option A (Direct MuJoCo Load)**:
   MuJoCo can load the URDF directly in Python:
   ```python
   import mujoco
   model = mujoco.MjModel.from_xml_path("models/custom_arm/urdf/custom_arm.urdf")
   ```

2. **Option B (Compile to Clean MJCF XML)**:
   Run the MuJoCo compiler utility to convert URDF to XML:
   ```powershell
   python -c "import mujoco; m = mujoco.MjModel.from_xml_path('models/custom_arm/urdf/custom_arm.urdf'); mujoco.mj_saveLastXML('models/custom_arm/scene.xml', m)"
   ```

3. Add the `<site name="attachment_site" pos="0 0 0.05"/>` tag inside your `hand gripper` body so the **Cartesian Inverse Kinematics (`ik_solver.py`)** automatically tracks your gripper tip!

---

## 5. Integrating with Phase 4 Code

Once the URDF/MJCF file is exported, seamlessly switch the simulator from the KUKA arm to your custom arm:

1. In [construction_executor.py](file:///d:/robot/cai_safety_checker%20-%20Phase%204/construction_executor.py#L32):
   ```python
   # Change:
   XML_PATH = Path(__file__).parent / "models" / "custom_arm" / "scene.xml"
   ```
2. The **Constitutional AI safety layer, the Cartesian IK solver, and the Web Dashboard will immediately control your team's custom arm!**

---

## 6. Checklist for Member 3

- [ ] Open `Assem1.SLDASM` in SolidWorks and verify all mates are resolved without warnings.
- [ ] Assign lightweight material densities (e.g., Aluminum 6061 or PLA/PETG) to calculate accurate link masses.
- [ ] Export URDF package with STL meshes.
- [ ] Place `custom_arm` folder under `models/` in Phase 4.
- [ ] Add `attachment_site` to end-effector and test in MuJoCo.
