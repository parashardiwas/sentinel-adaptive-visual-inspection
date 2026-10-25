# SOP-HAZ-04: Standard Operating Procedure for Food Grade Nut Sorting & Quality Assurance

## 1. Scope & Product Description
- **Category**: `hazelnut`
- **Product**: Whole shelled raw and roasted hazelnuts (Corylus avellana).
- **Standards Compliance**: USDA / UNECE Standard DDP-03 for Unshelled and Shelled Hazelnuts, Codex Alimentarius CXS 152.
- **Inspection Station**: Multi-angle diffuse dome lighting, food-grade belt conveyor with high-speed air-ejection nozzles.

---

## 2. Defect Taxonomy & Remediation Guidelines

### 2.1 Defect Code: `hole` (Insect Damage / Boring)
- **Visual Signature**: Circular cavity or tunnel entrance ($\ge 0.5$ mm diameter) through nut pericarp or kernel.
- **Root Cause**:
  1. Infestation by Curculio nucum (hazelnut weevil) larvae during orchard development.
  2. Sub-standard pre-storage fumigation or post-harvest screening.
- **Severity**: Critical / Reject (Food safety violation; zero-tolerance for commercial retail).
- **Disposition / Action**: **Reject & Quarantine**. Divert to non-food industrial oil extraction or incineration.
- **Protocol**: If $> 2\%$ of samples in a 500-unit lot exhibit `hole`, isolate warehouse lot for phytosanitary testing.

### 2.2 Defect Code: `crack` (Pericarp / Shell Fracture)
- **Visual Signature**: Structural crack or fissure splitting the kernel body, often with loose kernel fragments.
- **Root Cause**:
  1. Excessive mechanical impact during mechanical hulling or pneumatic conveying.
  2. Over-drying in thermal dehydration kilns ($> 42^\circ\text{C}$ causing brittle stress cracking).
- **Severity**: Moderate to High (Accelerated lipid oxidation, rancidity, and shelf-life degradation).
- **Disposition / Action**: **Downgrade to Confectionery / Chopped Kernel Processing**. Not acceptable for premium whole-nut retail packaging.

### 2.3 Defect Code: `cut` (Mechanical Scission or Blade Gouge)
- **Visual Signature**: Sharp, linear incision across nut surface caused by shearing contact with processing blades.
- **Root Cause**:
  1. De-stoning machine clearance too tight.
  2. Damaged rotor blades in cracking drum.
- **Severity**: Moderate.
- **Disposition / Action**: Re-route to diced hazelnut paste/praline processing line. Secondary antimicrobial wash required.

### 2.4 Defect Code: `print` (Foreign Ink / Branding Stamp Misplacement)
- **Visual Signature**: Inky residue, dark spotting, or conveyor belt transfer marks on kernel surface.
- **Root Cause**:
  1. Residue transfer from contaminated sorting belt or ink-jet packaging date coders.
  2. Solvent leaching from rubber transport guides.
- **Severity**: High (Chemical contamination risk).
- **Disposition / Action**: **Immediate Scrap**. Stop packaging line and inspect belt wash station.

---

## 3. RL Scheduler Configuration for Hazelnut Sorting
- **Low Risk ($\Delta S < -0.10$)**: Action `skip`. Bulk throughput up to 300 units/second.
- **Medium Risk ($-0.10 \le \Delta S \le +0.10$)**: Action `prediction`. High-speed air ejector activates if model votes defect (`prediction = 1`).
- **High Risk ($\Delta S > +0.10$)**: Action `expert`. Automatic sample diversion to QA lab for moisture analysis and peroxide value titration.
