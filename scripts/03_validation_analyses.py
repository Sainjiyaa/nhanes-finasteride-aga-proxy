from pathlib import Path
import os
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("NHANES_RAW_DIR", ROOT / "data" / "raw"))
OUT = ROOT / "outputs" / "derived"
AGGREGATE_OUT = ROOT / "outputs" / "aggregate"
AGGREGATE_OUT.mkdir(parents=True, exist_ok=True)

cohort = pd.read_csv(OUT / "finasteride_male_audit.csv")

criterion = cohort.loc[
    cohort["cycle_start"].ge(2013) & (cohort["reason_hair"] | cohort["reason_prostate"])
].copy()
criterion["target_hair"] = criterion["reason_hair"]

metrics = []
for cutoff in [35, 40, 45, 50, 53]:
    pred = criterion["RIDAGEYR"].lt(cutoff)
    y = criterion["target_hair"]
    tp = int((pred & y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    fp = int((pred & ~y).sum())
    row = {"cutoff": f"<{cutoff}", "tp": tp, "fn": fn, "tn": tn, "fp": fp}
    for name, num, den in [
        ("sensitivity", tp, tp + fn),
        ("specificity", tn, tn + fp),
        ("ppv", tp, tp + fp),
        ("npv", tn, tn + fn),
        ("agreement", tp + tn, len(criterion)),
    ]:
        row[name] = num / den if den else None
    metrics.append(row)
metrics = pd.DataFrame(metrics)
metrics.to_csv(OUT / "age_threshold_performance.csv", index=False)
metrics.to_csv(AGGREGATE_OUT / "age_threshold_performance.csv", index=False)

concurrent = []
adult = cohort["RIDAGEYR"].ge(18)
for flag in ["class_strict", "class_l989_age"]:
    for outcome in ["bph_specific", "ambiguous", "any_broad"]:
        retained = cohort[flag] & adult
        excluded = ~cohort[flag] & adult
        a = int((retained & cohort[outcome]).sum())
        b = int((retained & ~cohort[outcome]).sum())
        c = int((excluded & cohort[outcome]).sum())
        d = int((excluded & ~cohort[outcome]).sum())
        concurrent.append({
            "proxy": flag, "outcome": outcome,
            "retained_yes": a, "retained_no": b,
            "excluded_yes": c, "excluded_no": d,
        })
concurrent = pd.DataFrame(concurrent)
concurrent.to_csv(OUT / "concurrent_medication_tests.csv", index=False)
concurrent.to_csv(AGGREGATE_OUT / "concurrent_medication_tests.csv", index=False)

cycles = [
    ("1999-2000", ""), ("2001-2002", "_B"), ("2003-2004", "_C"),
    ("2005-2006", "_D"), ("2007-2008", "_E"), ("2009-2010", "_F"),
    ("2011-2012", "_G"), ("2013-2014", "_H"), ("2015-2016", "_I"),
    ("2017-2018", "_J"),
]
mins = []
for cycle, suffix in cycles:
    rx = pd.read_sas(RAW / f"RXQ_RX{suffix}.xpt", format="xport", encoding="latin1")
    drug = "RXD240B" if "RXD240B" in rx else "RXDDRUG"
    rx[drug] = rx[drug].astype(str).str.strip().str.upper()
    m = rx.loc[rx[drug].str.contains("MINOXIDIL", na=False)].copy()
    for _, r in m.iterrows():
        mins.append({
            "SEQN": r["SEQN"], "cycle": cycle, "drug": r[drug],
            "codes": "|".join(str(r.get(c, "")).strip() for c in ["RXDRSC1","RXDRSC2","RXDRSC3"]
                              if c in m.columns).strip("|"),
            "descriptions": "|".join(str(r.get(c, "")).strip() for c in ["RXDRSD1","RXDRSD2","RXDRSD3"]
                                     if c in m.columns).strip("|"),
        })
mins = pd.DataFrame(mins)
mins.to_csv(OUT / "minoxidil_inventory.csv", index=False)

print("Criterion n and classes")
print(len(criterion), criterion["target_hair"].value_counts().to_dict())
print(metrics.to_string(index=False))
print("\nConcurrent medication tests")
print(concurrent.to_string(index=False))
print("\nMinoxidil inventory")
print(mins.to_string(index=False))
