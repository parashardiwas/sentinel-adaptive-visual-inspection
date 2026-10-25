# SOP-GRD-02: Standard Operating Procedure for Industrial Mesh & Grid Inspection

## 1. Scope & Product Description
- **Category**: `grid`
- **Component**: Precision woven wire mesh, acoustic grilles, and structural micro-grids.
- **Material**: 316L Stainless Steel wire / electroplated mild steel.
- **Inspection Station**: High-angle coaxial LED illumination, 224x224 normalized FOV.

---

## 2. Defect Taxonomy & Visual Signatures

### 2.1 Defect Code: `bent` (Bent Wire Strand)
- **Visual Signature**: Localized curvature or pitch displacement in horizontal or vertical mesh strands. Loss of geometric periodicity.
- **Root Cause**:
  1. Tensioner roller misalignment in the wire weaving head.
  2. Mechanical collision with part ejector fingers during conveyor transfer.
  3. Excessive feed spool drag creating uneven strand tension.
- **Severity**: High (Impairs acoustic impedance and EMI shielding).
- **Rework Procedure**: Rework is **prohibited**. Part cannot be mechanically straightened without introducing work-hardening fractures. **Scrap part immediately**.
- **Preventive Maintenance**: Check guide roller bearings on Weave Line #3; inspect tensioner load cells.

### 2.2 Defect Code: `broken` (Fractured Wire Strand)
- **Visual Signature**: Discontinuous strand with visible gap; wire ends separated or retracted.
- **Root Cause**:
  1. Tensile stress exceeding yield strength during the cross-weave crimping stage.
  2. Wire inclusions or metallurgical micro-voids in raw wire stock.
  3. Worn ceramic eyelets on wire guides creating localized notch fatigue.
- **Severity**: Critical (Potential particulate shedding and complete functional failure).
- **Rework Procedure**: **Scrap part**. Isolate the entire batch for metallurgical batch coupon testing.
- **Preventive Maintenance**: Replace ceramic feed eyelets; review wire supplier tensile test certs.

### 2.3 Defect Code: `glue` (Adhesive Contamination)
- **Visual Signature**: Semi-transparent or amber residue bridging mesh apertures; localized specular reflection.
- **Root Cause**:
  1. Excessive dispensing pressure on upstream perimeter gasket applicator.
  2. Nozzle drool or inadequate vacuum pullback on adhesive metering valve.
  3. Insufficient curing oven exhaust causing adhesive vapor condensation.
- **Severity**: Moderate to High.
- **Rework Procedure**:
  1. Transfer unit to Rework Station B.
  2. Apply isopropyl alcohol (IPA 99.7%) via lint-free micro-swab.
  3. Ultrasonic immersion bath for 120 seconds at 40 kHz.
  4. Re-inspect through Vision Anomaly Detector. If score drops below 0.85, return to nominal stream.
- **Preventive Maintenance**: Clean adhesive dispense nozzles; recalibrate pressure transducer to 2.2 bar.

### 2.4 Defect Code: `metal_contamination` (Foreign Metal Particulate)
- **Visual Signature**: Foreign metallic swarf, chips, or weld spatter adhered to grid surface.
- **Root Cause**:
  1. Upstream laser cutting or spot-welding splatter without adequate suction shielding.
  2. Mechanical wear on overhead conveyor trolley bearings.
- **Severity**: Critical (Direct short-circuit risk in electronic enclosures).
- **Rework Procedure**:
  1. High-pressure ionized air blow-off (6 bar) in cleanroom booth.
  2. If foreign particle is sintered or welded, part is non-salvageable: **Scrap**.
- **Preventive Maintenance**: Check overhead conveyor drip trays and vacuum shroud on station #1.

### 2.5 Defect Code: `thread` (Textile / Fiber Contamination)
- **Visual Signature**: Elongated, non-conductive fibrous strand entangled across grid aperture.
- **Root Cause**:
  1. Shedding from operator PPE gloves or conveyor wipe-down cloths.
  2. Ambient particulate ingress due to negative cleanroom differential pressure.
- **Severity**: Low to Moderate.
- **Rework Procedure**: Automated HEPA vacuum pickup or anti-static air knife pass. 100% salvageable upon re-cleaning.

---

## 3. RL Scheduler Decision Matrix for Grid Line
- **Low Risk ($\Delta S < -0.10$)**: Action `skip`. Average throughput target: 120 units/min.
- **Medium Risk ($-0.10 \le \Delta S \le 0.10$)**: Action `prediction` or `expert`. Check for adhesive drool or loose fiber.
- **High Risk ($\Delta S > 0.10$)**: Action `expert`. Immediate routing to audit bin. If 5 consecutive units exhibit `bent` or `broken`, trigger **Line Stop Alarm (E-04)** for weaver tooling inspection.
