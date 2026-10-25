# SOP-LTH-05: Standard Operating Procedure for Automotive & Upholstery Leather Quality Control

## 1. Scope & Product Description
- **Category**: `leather`
- **Product**: Semi-aniline and full-grain finished bovine upholstery hides for luxury automotive seating and premium goods.
- **Inspection Station**: Calibrated D65 daylight simulator booth with diffuse dome illumination and surface profilometer scanner.

---

## 2. Defect Taxonomy & Remediation Guidelines

### 2.1 Defect Code: `cut` (Open Surface Laceration)
- **Visual Signature**: Linear penetration through topcoat and grain layer into the dermis fiber network.
- **Root Cause**:
  1. Barbed wire scars or butcher flaying scissions on raw rawhide.
  2. CNC nesting cutter blade over-travel or un-retracted blade during rapid traverse.
- **Severity**: Critical. Structural integrity destroyed.
- **Rework Procedure**: Rework prohibited. **Scrap panel**. Flag coordinate in CNC digital nesting software to exclude defective patch.

### 2.2 Defect Code: `poke` (Puncture / Perforation Void)
- **Visual Signature**: Circular or irregular localized puncture hole penetrating into leather substructure.
- **Root Cause**:
  1. Tenter frame pinning tears during drying stage.
  2. Handling punctures by pallet fork tines or transport hooks.
- **Severity**: Critical.
- **Rework Procedure**: **Scrap panel**. Update nested cutting template.

### 2.3 Defect Code: `fold` (Crease / Structural Wrinkle)
- **Visual Signature**: Deep, irreversible surface fold line with localized grain distortion or micro-cracking.
- **Root Cause**:
  1. Improper folding during bulk palletized shipping instead of roll packaging.
  2. Uneven vacuum drying plate pressure distribution.
- **Severity**: Moderate to High.
- **Rework Procedure**:
  1. Thermal stretching on heated iron plate at $65^\circ\text{C}$ with conditioning steam mist.
  2. Re-measure surface flatness using profilometer. If crease depth $< 0.05$ mm, pass as Grade B. Otherwise, down-sort to sub-assembly components.

### 2.4 Defect Code: `glue` (Adhesive Spotting / Overspray)
- **Visual Signature**: Raised, tacky, or glossy adhesive droplet on leather grain.
- **Root Cause**:
  1. Lamination adhesive overspray from adjacent foam bonding station.
  2. Fingerprint contamination during manual positioning.
- **Severity**: Moderate.
- **Rework Procedure**:
  1. Apply specialized leather cleaner (pH neutral, water-based).
  2. Gentle mechanical agitation with natural crepe rubber block.
  3. Condition with carnauba-based leather protective balm. Re-score anomaly.

### 2.5 Defect Code: `color` (Shade Variation / Pigment Bleed)
- **Visual Signature**: Localized tint deviation, blotchiness, or $\Delta E > 0.8$ against master spectrophotometer standard.
- **Root Cause**:
  1. Inadequate spray gun atomization in automatic finishing carousel.
  2. Batch-to-batch dye lot variance.
- **Severity**: Moderate (Aesthetic non-conformance).
- **Rework Procedure**: Secondary tint coat application in manual refinishing booth, followed by clear polyurethane protective seal.

---

## 3. RL Scheduler Decision Rules for Leather Inspection
- **Low Risk ($\Delta S < -0.10$)**: Action `skip`. Panel advances to automated waterjet cutting cell.
- **Medium Risk ($-0.10 \le \Delta S \le +0.10$)**: Action `prediction`. Automated marking pen highlights zone on hide.
- **High Risk ($\Delta S > +0.10$)**: Action `expert`. Master leather craftsman manual audit and grade classification (Grade A vs. Grade B vs. Scrap).
