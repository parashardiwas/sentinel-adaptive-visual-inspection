# SOP-TB-06: Standard Operating Procedure for Oral Hygiene Product Assembly & Inspection

## 1. Scope & Product Description
- **Category**: `toothbrush`
- **Product**: Injection-molded multi-tufted manual toothbrushes with nylon-6,12 filaments.
- **Compliance Standard**: ISO 20126 (Dentistry — Manual toothbrushes — General requirements and test methods).
- **Inspection Station**: Dual-axis orthogonal telecentric inspection chamber with high-speed strobe backlight.

---

## 2. Defect Taxonomy & Remediation Guidelines

### 2.1 Defect Code: `defective` (Tufting Non-Conformance / Filament Malformation)
- **Visual Sub-types**:
  1. **Splayed Bristles**: Peripheral nylon filaments bent outwards $> 15^\circ$ from vertical tuft axis.
  2. **Missing Tuft / Empty Hole**: Failure of anchor wire insertion leaving an open anchor hole.
  3. **Uneven Trim / Long Filament**: Individual filament protruding $> 0.5$ mm above nominal crown contour.
  4. **Cracked Handle Head**: Micro-fissures around tufting anchor holes caused by excessive punching force.
- **Root Cause Analysis**:
  1. *Splayed Bristles*: Dull end-rounding rotary cutter knives or thermal overheating during abrasive polishing.
  2. *Missing Tuft*: Jammed brass anchor wire feed spool or empty filament picker cassette.
  3. *Cracked Head*: High insertion tool velocity on polypropylene head substrate under cold ambient shop floor conditions ($< 18^\circ\text{C}$).
- **Severity**: Critical (Potential oral soft-tissue abrasion or choking hazard from loose bristle detachment).
- **Rework Procedure**: Rework is strictly **prohibited** under medical device and hygiene standards. **100% Scrap**.
- **Preventive Maintenance Action**:
  - Stop tufting carousel #2.
  - Replace end-rounding abrasive discs every 80,000 cycles.
  - Verify anchor wire tension (nominal: 18 N pull-out retention force).

---

## 3. RL Scheduler Decision Policy for Toothbrush Lines
- **Low Risk ($\Delta S < -0.10$)**: Action `skip`. Conveyor line rate: 180 brushes/minute.
- **Medium Risk ($-0.10 \le \Delta S \le +0.10$)**: Action `prediction`. High-speed optical reject cylinder ejects brush to scrap bin.
- **High Risk ($\Delta S > +0.10$)**: Action `expert`. Line technician pulls 10 consecutive brushes for manual pull-force retention testing.
