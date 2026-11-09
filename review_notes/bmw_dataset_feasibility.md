# BMW i3 dataset feasibility audit

**Status: NO-GO for training until Measurement Data are locally available and inspected.**

Target dataset: *Battery and Heating Data in Real Driving Cycles*  
Steinstraeter, Buberger, Trifonov (2020)  
DOI: https://doi.org/10.21227/6JR9-5235  
IEEE DataPort: https://ieee-dataport.org/open-access/battery-and-heating-data-real-driving-cycles

## Why training has not started

1. The Measurement Data archive is **not present** in this workspace.
2. IEEE DataPort requires **login to download** `Measurement Data.zip` (~35.7 MB).
3. Without the actual CSV/MAT files, trip usability, SoC quality, current sign, charging segments, and summer/winter completeness cannot be verified scientifically.
4. Fabricating measurements or substituting simulation outputs would violate the review protocol.

**Blocker:** Author must download Measurement Data locally (do not commit the zip) and place it under a gitignored path such as `data/bmw_i3/` or `datasets/bmw_i3/`.

## Published dataset summary (pre-download)

| Item | Published claim |
|---|---|
| Vehicle | BMW i3 (60 Ah) |
| Trips | 72 real driving trips |
| Contents | Environment, vehicle, battery (V, I, T, SoC), heating circuit |
| Formats | CSV and MAT |
| License | Creative Commons Attribution |
| Related paper | Electrothermal recuperation / peak-power shaving studies |

Known caveats from the literature and community discussion (to verify on the files):

- Summer records are often incomplete relative to winter records.
- TripB04 has a reported stationary-charging issue and must not be trusted blindly.
- Auxiliary heating/AC load is large; net battery power ≠ traction power alone.
- Current sign convention must be verified against voltage and SoC change before defining measured energy.

## Feature compatibility vs HELECAR-D (expected)

| HELECAR-D PINN input | BMW expected availability | Notes |
|---|---|---|
| speed | Likely yes | Vehicle speed reported |
| acceleration | Likely yes / derivable | Longitudinal acceleration listed in secondary papers |
| grade / elevation | Elevation yes → grade derivable | Must audit GPS elevation quality |
| temperature | Ambient / battery / cabin | Use ambient vehicle temperature carefully |
| humidity | Unknown / likely missing | Common missing HELECAR feature |
| wind speed | Unknown / likely missing | Do not invent |
| traffic | Missing | Do not invent |
| speed limit | Missing | Do not invent |
| SoC | Yes | Supervision only; never a power-branch input |
| battery V, I | Yes | **Evaluation only**; never for PINN training targets or inputs |

If humidity / wind / traffic / speed limit are absent, a feature-adapted replication is allowed only if:

- the same reduced feature set is used for PINN and Data-Driven MLP;
- the experiment is labeled as feature-adapted, not identical to HELECAR-D;
- missing features are not replaced by constants chosen after seeing test results.

## Physical parameters (sources required; not measured here)

Any BMW i3 physics YAML must cite sources and mark assumptions. Provisional targets to document after download:

| Parameter | Typical literature / OEM range | Status |
|---|---|---|
| Mass | ~1195–1345 kg depending on curb + payload assumption | Assumption required |
| CdA | ~0.65–0.75 m² class for i3 | Assumption / OEM aero figures |
| Usable battery capacity | 60 Ah pack; usable energy ≈ 18–22 kWh class, **not** equal to nominal by default | Must examine; do not treat as exact ground truth |
| Residual bound L | Must be re-justified for the larger vehicle; not copied as 4 kW without rationale | Predetermined, not test-tuned |

## Independent energy reference

If V and I are trustworthy:

1. Verify current sign vs discharge / charge and vs SoC decrease.
2. Compute `P_batt = V * I` with the verified convention.
3. Integrate with actual Δt.
4. Compare integrated battery energy to SoC-derived energy under an assumed capacity.
5. Use measured energy **only for final evaluation**, never for weak supervision of the PINN.

## GO / NO-GO recommendation

| Criterion | Result |
|---|---|
| Local Measurement Data available | **NO** |
| File-level audit completed | **NO** |
| Enough independent driving trips after QC | Unknown |
| Credible SoC supervision | Unknown |
| Independent V·I energy reference | Unknown |
| Feature-adapted protocol defined | Drafted only |

### Recommendation

**NO-GO for Phase 4–5 BMW training at this time.**

Proceed with:

1. HELECAR-D ablations and sensitivity (Phases 2 and Additional Phase) — these are unblocked.
2. Author download of Measurement Data.
3. Then re-run this audit on the real files and revise the GO/NO-GO.

## Alternative if BMW remains unavailable

KU Leuven EV energy dataset: https://doi.org/10.48804/8KPDTW  

Same rule: audit before training; do not start training solely to produce another-vehicle numbers.
