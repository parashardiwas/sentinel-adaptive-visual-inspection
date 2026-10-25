# SOP-GEN-01: Standard Operating Procedure for Adaptive Visual Inspection & RL Scheduling

## 1. Document Control
- **Document ID**: SOP-GEN-01
- **Title**: Adaptive Visual Inspection Policy & Reinforcement Learning Scheduling Protocol
- **Classification**: Quality Assurance & Process Control Standard
- **Effective Version**: 2.4
- **Applicable Domains**: Precision Manufacturing, Textile Assembly, Food Processing, Upholstery

---

## 2. System Overview & Operational Architecture
The Adaptive Visual Inspection System integrates two intelligent components to balance inspection throughput with quality assurance:
1. **Computer Vision Anomaly Detector (Mahalanobis Distance on ResNet-18 Embeddings)**:
   - Measures the statistical Mahalanobis distance between product feature embeddings and the distribution of nominal (defect-free) reference units.
   - Generates a normalized anomaly score $S \in [0.0, 1.0]$.
   - Flags an anomaly whenever $S \ge S_{threshold}$ (where $S_{threshold} \approx 0.95$ corresponds to the 95th percentile of fault-free validation units).
2. **Reinforcement Learning Inspection Scheduler (Q-Learning Agent)**:
   - Evaluates the incoming product risk tier and selects an optimal inspection action.
   - Designed to eliminate 100% manual inspection bottlenecks while capping missed defect rates below 1.5%.

---

## 3. Risk Tier Classification
Products arriving on the conveyor are categorized into three operational risk tiers based on their threshold margin:
$$\Delta S = S_{normalized} - S_{threshold}$$

| Risk Tier | Condition | Operational Definition |
|---|---|---|
| **Low Risk** | $\Delta S < -0.10$ | Item score is well below anomaly threshold. High statistical certainty of being nominal. |
| **Medium Risk** | $-0.10 \le \Delta S \le +0.10$ | Item is near the decision boundary. Ambiguous visual features; potential false alarm or subtle micro-defect. |
| **High Risk** | $\Delta S > +0.10$ | Item score is significantly above threshold. High probability of genuine defect. |

---

## 4. Reinforcement Learning Action Protocol

The Q-learning scheduler decides one of three actions per unit:

### 4.1 Action: `skip` (Bypass Inspection)
- **Simulated Execution Cost**: \$0.00
- **Operational Rule**: Product bypasses secondary review and continues directly down the production line.
- **Permissible Conditions**: Typically selected for **Low Risk** items when the upstream defect rate is stable ($\le 3\%$).
- **Failure Penalty**: If a defective unit is skipped, a critical escape penalty (\$50 to \$100) is assessed for customer escape.

### 4.2 Action: `prediction` (Automated Model Gating)
- **Simulated Execution Cost**: \$1.00 (Compute, actuation, and automated sorting overhead)
- **Operational Rule**: The system relies strictly on the vision model's automated classification (`prediction = 1` vs `0`).
- **Permissible Conditions**: Preferred for moderate confidence items where automated diversion to a secondary scrap bin is sufficient.

### 4.3 Action: `expert` (Human Expert Routing / Secondary Station)
- **Simulated Execution Cost**: \$5.00 (Manual labor, high-resolution microscope inspection, tactile audit)
- **Operational Rule**: Item is diverted off the main line to a dedicated quality audit station. Assumes 100% human ground-truth verification.
- **Permissible Conditions**: Triggered when uncertainty is elevated (**Medium Risk**) or during suspected batch drift / defect spikes (**High Risk**).

---

## 5. Drift Detection & Adaptation Protocols
When the RL agent detects a sustained spike in high-risk classifications (e.g., > 10% high-risk items over a rolling window of 100 units):
1. **Dynamic Shift**: The agent transitions into conservative mode, temporarily increasing the `expert` action selection rate to 70–90% for that category.
2. **Maintenance Interlock**: If the expert confirmation rate exceeds 15% confirmed defects over 200 consecutive cycles, an automated maintenance ticket is generated, and the cell operator must execute the category-specific root-cause inspection protocol.
