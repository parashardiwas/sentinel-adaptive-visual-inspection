# SOP-ZIP-03: Standard Operating Procedure for Zipper & Fastener Quality Control

## 1. Scope & Product Description
- **Category**: `zipper`
- **Component**: Continuous coiled and molded-tooth fastener chains with woven side tapes.
- **Material**: Polyester tape with polyacetal (POM) molded teeth and nylon spiral elements.
- **Inspection Station**: Dual linear telecentric optical scanner, multi-channel side darkfield illumination.

---

## 2. Defect Taxonomy & Remediation Protocol

### 2.1 Defect Code: `broken_teeth` (Fractured Fastener Element)
- **Visual Signature**: Complete absence or transverse fracture of one or more interlocking teeth.
- **Root Cause**:
  1. Excessive injection molding clamp tonnage shearing tooth neck.
  2. Inadequate mold cooling cycle leading to brittle sprue detachment.
  3. Physical jam in chain intermeshing gear.
- **Severity**: Critical (Mechanical failure; non-functional fastener).
- **Rework Procedure**: **Scrap**. Salvage of broken tooth sections is strictly prohibited.
- **Machine Adjustment**: Inspect injection mold core pins and verify ejection pin timing.

### 2.2 Defect Code: `split_teeth` (Longitudinally Split or Delaminated Tooth)
- **Visual Signature**: Partial fissure running along the central axis of the fastener tooth.
- **Root Cause**:
  1. Incomplete resin plasticization or thermal degradation during melt extrusion.
  2. Moisture entrapment in hygroscopic raw POM polymer granules.
- **Severity**: High. Tooth will yield under lateral tensile proof testing (ASTM D2061).
- **Rework Procedure**: **Scrap**. Tag batch for raw material moisture content re-audit.

### 2.3 Defect Code: `squeezed_teeth` (Deformed / Crushed Pitch)
- **Visual Signature**: Flattened tooth profile with reduced pitch spacing; tooth tip lateral deformation.
- **Root Cause**:
  1. Over-tightening of pinch drive rollers downstream of molding head.
  2. Chain misalignment entering the cooling flume.
- **Severity**: High (Will cause slider stall or burst under load).
- **Rework Procedure**: **Scrap section**. Adjust downstream nip roller clearance to 1.85 mm $\pm 0.05$ mm.

### 2.4 Defect Code: `rough` (Surface Flashing / Burr)
- **Visual Signature**: Irregular polymer excess along parting line of the tooth edge.
- **Root Cause**:
  1. Wear on mold parting surfaces.
  2. Injection pressure overshoot exceeding 140 bar.
- **Severity**: Moderate (Aesthetic issue and rough slider action).
- **Rework Procedure**: Rotary thermal de-flashing or secondary trimming pass. Unit can be returned to inventory if burr is removed without dimension loss.

### 2.5 Defect Code: `fabric_border` & `fabric_interior` (Tape Weave Irregularities)
- **Visual Signature**:
  - `fabric_border`: Frayed selvage, loose loop, or wavy border along tape edge.
  - `fabric_interior`: Missing warp/weft yarn, pulled thread, or needle perforation in tape body.
- **Root Cause**:
  1. Worn heddle wires on needle loom.
  2. Yarn tension imbalance between right and left selvedge bobbins.
- **Severity**: Moderate to High (Risk of slider run-off or tape tearing during garment stitching).
- **Rework Procedure**:
  - If frayed edge is $< 2$ mm, heat-seal edge with ultrasonic cutter.
  - If interior yarn is broken across $> 3$ threads, **Scrap**.

---

## 3. RL Scheduler Operational Rules for Zipper Line
- **Low Risk ($\Delta S < -0.10$)**: Action `skip`. Full conveyor velocity ($1.2$ m/s).
- **Medium Risk ($-0.10 \le \Delta S \le +0.10$)**: Action `prediction`. Automated pneumatic diverter separates suspect lengths.
- **High Risk ($\Delta S > +0.10$)**: Action `expert`. Line operator inspects tooth pitch gauge. If `broken_teeth` confirmed, inspect injection molding station immediately.
