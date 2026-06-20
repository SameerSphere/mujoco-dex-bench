# Task Documentation

Detailed specifications for each environment in the MuJoCo Dexterous Hand Benchmark.

---

## DexPenSpin-v0

**Physics:** A low-inertia capsule pen (mass 12 g) is held above the palm. Finger tendons couple PIP/DIP joints with ratio 0.6, requiring coordinated torques to maintain grasp while applying spin.

**Reward:**

\[
r = -|\theta_{\text{spin}} - \pi/2| + \mathbb{1}_{\text{success}} \cdot 10 - 0.01 \|a\|^2
\]

where \(\theta_{\text{spin}}\) is rotation about the pen long axis.

**Success:** Rotation ≥ 15° about long axis from initial orientation.

**Training challenges:** Slip during high angular velocity; balance grasp force vs. spin torque.

**Suggested hyperparameters:** SAC `buffer_size=100k`, `batch_size=256`, `learning_rate=3e-4`, domain randomisation enabled after 500k steps.

---

## DexKeyInsert-v0

**Physics:** Thin key (8×40×3 mm) must align with a lock keyhole site. Tight lateral clearance demands precise 6-DoF control.

**Reward:**

\[
r = -(w_p \|p - p^*\| + w_r \Delta q) + 0.5 \cdot \mathbb{1}_{\text{contact}} + \mathbb{1}_{\text{success}} \cdot 10 - 0.01 \|a\|^2
\]

**Success:** Position ≤ 2 mm, orientation ≤ 5°.

**Training challenges:** Sparse contact signal early in training; key tumbling.

**Suggested hyperparameters:** HER with `n_sampled_goal=4`, higher `pos_weight=2.0`.

---

## DexInHandReorient-v0

**Physics:** 35 mm cube with uniform inertia held in palm. No table support — object must stay in hand.

**Reward:** Standard dense pose distance with 5 mm / 10° tolerances.

**Success:** Position ≤ 5 mm, orientation ≤ 10° relative to sampled goal.

**Training challenges:** Dropping object; local minima with partial rotations.

---

## DexChopstickGrasp-v0

**Physics:** Two independent capsule chopsticks (4 mm × 200 mm) must be coordinated to lift an object to target height.

**Reward:** Position-focused dense shaping to target at \(z = 0.25\) m.

**Success:** 10 mm position, 20° orientation.

**Training challenges:** Bimanual-style coordination with single hand; stick slip.

---

## DexBallRoll-v0

**Physics:** 50 mm sphere on table with rolling friction. Smooth contacts favour fingertip rolling strategies.

**Reward:**

\[
r = -\|p - p^*\| + \mathbb{1}_{\text{success}} \cdot 10 - 0.01 \|a\|^2
\]

(orientation ignored)

**Success:** Position ≤ 8 mm.

**Training challenges:** Easiest task — useful for pipeline validation.

**Suggested hyperparameters:** PPO or SAC, 500k–1M steps sufficient.

---

## DexPegInHole-v0

**Physics:** Cylinder peg (16 mm diameter) into hole with ~2 mm clearance. Contact reward when peg touches plate.

**Reward:** Dense pose + contact bonus + success bonus.

**Success:** 1.5 mm, 3°.

**Training challenges:** Classic insertion search; jamming.

---

## DexDoorOpen-v0

**Physics:** 0.4 kg door panel on hinge with spring stiffness 0.5 N·m/rad. Handle site on panel face.

**Reward:**

\[
r = -|\theta_{\text{door}} - 60°|
\]

**Success:** Door hinge angle ≥ 60°.

**Training challenges:** Large workspace motion; sustained pulling force.

---

## DexScissorsCut-v0

**Physics:** Two blade geoms on scissor hinge; soft paper geom with compliant contact (`solref`, `solimp`).

**Reward:** Contact-based — reward when both blades touch paper.

**Success:** Simultaneous blade–paper contact.

**Training challenges:** Tool use; hinge actuation not directly controlled (free scissors body).

---

## DexCardFlip-v0

**Physics:** Ultra-thin card (0.8 mm) with fragile contact parameters. Requires delicate normal forces.

**Reward:**

\[
r = -\|p - p^*\| - |\theta_{\text{flip}} - \pi|
\]

**Success:** Position ≤ 5 mm, flip ≥ 170° from start orientation.

**Training challenges:** Card bending, edge contacts, extreme aspect ratio.

---

## Shared Observation Vector

| Component | Dim |
|-----------|-----|
| sin(joint pos) | 15 |
| cos(joint pos) | 15 |
| joint velocity | 15 |
| fingertip touch | 5 |
| object pose | 7 |
| object velocity | 6 |
| palm → object | 3 |
| **Total** | **66** |
