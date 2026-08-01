from pathlib import Path
import math
import os
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("NHANES_RAW_DIR", ROOT / "data" / "raw"))
OUT = ROOT / "outputs" / "derived"
OUT.mkdir(exist_ok=True)

CYCLES = [
    ("1999-2000", "", 1999),
    ("2001-2002", "_B", 2001),
    ("2003-2004", "_C", 2003),
    ("2005-2006", "_D", 2005),
    ("2007-2008", "_E", 2007),
    ("2009-2010", "_F", 2009),
    ("2011-2012", "_G", 2011),
    ("2013-2014", "_H", 2013),
    ("2015-2016", "_I", 2015),
    ("2017-2018", "_J", 2017),
]

PROSTATE_FILES = {
    1999: ("KIQ.xpt", ["KIQ120", "KIQ200"]),
    2001: ("KIQ_P_B.xpt", ["KIQ106", "KIQ121", "KID182"]),
    2003: ("KIQ_P_C.xpt", ["KIQ106", "KIQ121", "KIQ182"]),
    2005: ("KIQ_P_D.xpt", ["KIQ490", "KIQ121", "KIQ182"]),
    2007: ("KIQ_P_E.xpt", ["KIQ490", "KIQ121", "KIQ182"]),
}

HAIR_CODES = {"L64", "L65.9"}
PROSTATE_CODES = {"N40", "N42.9", "C61"}
BPH_SPECIFIC = {
    "TAMSULOSIN", "TAMSULOSIN HYDROCHLORIDE", "ALFUZOSIN", "SILODOSIN", "DUTASTERIDE",
    "DUTASTERIDE AND TAMSULOSIN",
}
AMBIGUOUS = {
    "DOXAZOSIN", "DOXAZOSIN MESYLATE", "TERAZOSIN",
    "TERAZOSIN HYDROCHLORIDE", "TADALAFIL",
}


def read_xpt(name):
    return pd.read_sas(RAW / name, format="xport", encoding="latin1")


def drug_col(df):
    return "RXD240B" if "RXD240B" in df else "RXDDRUG"


people = []
all_drugs = []
for cycle, suffix, start in CYCLES:
    rx = read_xpt(f"RXQ_RX{suffix}.xpt")
    dc = drug_col(rx)
    rx[dc] = rx[dc].astype(str).str.strip().str.upper()
    rx["cycle"] = cycle
    rx["cycle_start"] = start
    all_drugs.append(rx[["SEQN", dc, "cycle", "cycle_start"]].rename(columns={dc: "drug"}))

    fin = rx.loc[rx[dc].eq("FINASTERIDE")].copy()
    demo = read_xpt(f"DEMO{suffix}.xpt")
    keep = [
        "SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "DMDEDUC2",
        "INDFMPIR", "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "WTMEC4YR",
    ]
    keep = [c for c in keep if c in demo]
    fin = fin.merge(demo[keep], on="SEQN", how="left")
    fin["cycle"] = cycle
    fin["cycle_start"] = start

    if start in PROSTATE_FILES:
        file_name, pvars = PROSTATE_FILES[start]
        p = read_xpt(file_name)[["SEQN"] + pvars]
        fin = fin.merge(p, on="SEQN", how="left")
        fin["prostate_any_positive"] = fin[pvars].eq(1).any(axis=1)
        fin["prostate_any_negative"] = fin[pvars].eq(2).any(axis=1)
        fin["prostate_all_missing"] = fin[pvars].isna().all(axis=1)
        fin["questionnaire_evidence"] = np.select(
            [fin["prostate_any_positive"], ~fin["prostate_any_positive"] & fin["prostate_any_negative"]],
            ["positive", "negative"],
            default="missing",
        )
    else:
        fin["prostate_any_positive"] = False
        fin["prostate_any_negative"] = False
        fin["prostate_all_missing"] = True
        fin["questionnaire_evidence"] = "not_available"

    if start >= 2013:
        reason_cols = [c for c in ["RXDRSC1", "RXDRSC2", "RXDRSC3"] if c in fin]
        for c in reason_cols:
            fin[c] = fin[c].astype(str).str.strip().str.upper().replace({"NAN": ""})
        fin["reason_codes"] = fin[reason_cols].agg("|".join, axis=1).str.strip("|")
        fin["reason_hair"] = fin[reason_cols].isin(HAIR_CODES).any(axis=1)
        fin["reason_prostate"] = fin[reason_cols].isin(PROSTATE_CODES).any(axis=1)
        fin["reason_l989"] = fin[reason_cols].eq("L98.9").any(axis=1)
    else:
        fin["reason_codes"] = ""
        fin["reason_hair"] = False
        fin["reason_prostate"] = False
        fin["reason_l989"] = False
    people.append(fin)

cohort = pd.concat(people, ignore_index=True, sort=False)
cohort = cohort.loc[cohort["RIAGENDR"].eq(1)].copy()

# Conservative primary reconstruction:
# 1999-2008: exclude if any prostate item is affirmative; otherwise retain a
# questionnaire-negative user, or an entirely unassessed user aged <40.
# 2009-2012: retain age <40.
# 2013-2018: retain only explicit L64/L65.9 hair-loss indications.
cohort["class_strict"] = False
cohort["class_l989_age"] = False
cohort["inclusion_reason"] = "excluded"

m_early = cohort["cycle_start"].le(2007)
m_neg = m_early & ~cohort["prostate_any_positive"] & cohort["prostate_any_negative"]
m_age_missing = m_early & cohort["prostate_all_missing"] & cohort["RIDAGEYR"].lt(40)
cohort.loc[m_neg | m_age_missing, ["class_strict", "class_l989_age"]] = True
cohort.loc[m_neg, "inclusion_reason"] = "questionnaire negative (no affirmative prostate item)"
cohort.loc[m_age_missing, "inclusion_reason"] = "age <40 with no prostate questionnaire data"

m_mid = cohort["cycle_start"].isin([2009, 2011]) & cohort["RIDAGEYR"].lt(40)
cohort.loc[m_mid, ["class_strict", "class_l989_age"]] = True
cohort.loc[m_mid, "inclusion_reason"] = "age <40 (no questionnaire or reason-for-use data)"

m_late_hair = cohort["cycle_start"].ge(2013) & cohort["reason_hair"]
cohort.loc[m_late_hair, ["class_strict", "class_l989_age"]] = True
cohort.loc[m_late_hair, "inclusion_reason"] = "explicit hair-loss reason-for-use"

# Manuscript variant that additionally admits the single L98.9 user by age.
m_l989 = cohort["cycle_start"].ge(2013) & cohort["reason_l989"] & cohort["RIDAGEYR"].lt(40)
cohort.loc[m_l989, "class_l989_age"] = True
cohort.loc[m_l989, "inclusion_reason"] = "L98.9 plus age <40 (composite proxy evidence)"

all_drugs = pd.concat(all_drugs, ignore_index=True)
co_meds = all_drugs.merge(cohort[["SEQN", "cycle", "class_strict", "class_l989_age"]], on=["SEQN", "cycle"])
co_meds = co_meds.loc[~co_meds["drug"].eq("FINASTERIDE")].copy()
flags = co_meds.groupby(["SEQN", "cycle"]).agg(
    bph_specific=("drug", lambda s: s.isin(BPH_SPECIFIC).any()),
    ambiguous=("drug", lambda s: s.isin(AMBIGUOUS).any()),
    med_list=("drug", lambda s: "|".join(sorted(set(s)))),
).reset_index()
cohort = cohort.merge(flags, on=["SEQN", "cycle"], how="left")
cohort[["bph_specific", "ambiguous"]] = (
    cohort[["bph_specific", "ambiguous"]].astype("boolean").fillna(False).astype(bool)
)
cohort["any_broad"] = cohort["bph_specific"] | cohort["ambiguous"]

# Add examination variables used in weighted descriptive analyses.
features = []
survey_people = []
for cycle, suffix, start in CYCLES:
    bmx = read_xpt(f"BMX{suffix}.xpt")[["SEQN", "BMXBMI"]]
    smq = read_xpt(f"SMQ{suffix}.xpt")[["SEQN", "SMQ020"]]
    x = bmx.merge(smq, on="SEQN", how="outer")
    x["cycle"] = cycle
    features.append(x)
    demo = read_xpt(f"DEMO{suffix}.xpt")
    dkeep = [
        "SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "DMDEDUC2",
        "INDFMPIR", "SDMVPSU", "SDMVSTRA", "WTMEC2YR", "WTMEC4YR",
    ]
    dkeep = [c for c in dkeep if c in demo]
    full = demo[dkeep].merge(bmx, on="SEQN", how="left").merge(smq, on="SEQN", how="left")
    full["cycle"] = cycle
    full["cycle_start"] = start
    survey_people.append(full)
features = pd.concat(features, ignore_index=True)
cohort = cohort.merge(features, on=["SEQN", "cycle"], how="left")
cohort["ever_smoked"] = cohort["SMQ020"].map({1.0: 1.0, 2.0: 0.0})
cohort["college_grad"] = cohort["DMDEDUC2"].map(
    {1.0: 0.0, 2.0: 0.0, 3.0: 0.0, 4.0: 0.0, 5.0: 1.0}
)

# Ten-cycle weight. 1999-2002 observations use the 4-year MEC weight and
# contribute 2/10; later 2-year cycles contribute 1/10.
cohort["WT20YR"] = np.where(
    cohort["cycle_start"].isin([1999, 2001]),
    cohort.get("WTMEC4YR", np.nan) * 0.2,
    cohort.get("WTMEC2YR", np.nan) * 0.1,
)

survey_people = pd.concat(survey_people, ignore_index=True, sort=False)
survey_people["WT20YR"] = np.where(
    survey_people["cycle_start"].isin([1999, 2001]),
    survey_people.get("WTMEC4YR", np.nan) * 0.2,
    survey_people.get("WTMEC2YR", np.nan) * 0.1,
)
survey_people["ever_smoked"] = survey_people["SMQ020"].map({1.0: 1.0, 2.0: 0.0})
survey_people["college_grad"] = survey_people["DMDEDUC2"].map({1.0: 0.0, 2.0: 0.0, 3.0: 0.0, 4.0: 0.0, 5.0: 1.0})
survey_people = survey_people.merge(
    cohort[["SEQN", "cycle", "class_strict", "class_l989_age",
            "questionnaire_evidence", "reason_hair"]],
    on=["SEQN", "cycle"], how="left",
)
survey_people[["class_strict", "class_l989_age"]] = (
    survey_people[["class_strict", "class_l989_age"]].astype("boolean").fillna(False).astype(bool)
)
survey_people["reason_hair"] = survey_people["reason_hair"].astype("boolean")
survey_people["adult"] = survey_people["RIDAGEYR"].ge(18)
survey_people["higher_certainty_strict"] = (
    survey_people["adult"] & survey_people["class_strict"] &
    ((survey_people["questionnaire_evidence"] == "negative") | survey_people["reason_hair"].fillna(False))
)
survey_people["strict_excluding_2009_2012"] = (
    survey_people["adult"] & survey_people["class_strict"] &
    ~survey_people["cycle_start"].isin([2009, 2011])
)
survey_people["primary_excluding_2009_2012"] = (
    survey_people["adult"] & survey_people["class_l989_age"] &
    ~survey_people["cycle_start"].isin([2009, 2011])
)
survey_people.to_csv(OUT / "full_survey_domain.csv", index=False)

cohort.to_csv(OUT / "finasteride_male_audit.csv", index=False)
co_meds.to_csv(OUT / "finasteride_concurrent_medications.csv", index=False)

rows = []
for period, mask in [
    ("1999-2008", cohort["cycle_start"].le(2007)),
    ("2009-2012", cohort["cycle_start"].isin([2009, 2011])),
    ("2013-2018", cohort["cycle_start"].ge(2013)),
    ("all", pd.Series(True, index=cohort.index)),
]:
    d = cohort.loc[mask]
    rows.append({
        "period": period,
        "male_finasteride_users": len(d),
        "strict_proxy_all_ages": int(d["class_strict"].sum()),
        "strict_proxy_adults": int((d["class_strict"] & d["RIDAGEYR"].ge(18)).sum()),
        "l989_age_proxy_all_ages": int(d["class_l989_age"].sum()),
        "l989_age_proxy_adults": int((d["class_l989_age"] & d["RIDAGEYR"].ge(18)).sum()),
        "prostate_any_positive": int(d["prostate_any_positive"].sum()),
        "questionnaire_negative_no_positive": int((~d["prostate_any_positive"] & d["prostate_any_negative"]).sum()),
        "questionnaire_missing": int(d["prostate_all_missing"].sum()),
        "explicit_hair": int(d["reason_hair"].sum()),
        "L98_9": int(d["reason_l989"].sum()),
        "explicit_prostate": int(d["reason_prostate"].sum()),
    })
cohort_counts = pd.DataFrame(rows)
cohort_counts.to_csv(OUT / "cohort_counts.csv", index=False)
aggregate_out = ROOT / "outputs" / "aggregate"
aggregate_out.mkdir(parents=True, exist_ok=True)
cohort_counts.to_csv(aggregate_out / "cohort_counts.csv", index=False)

print(cohort_counts.to_string(index=False))
print("\n1999-2008 mutually exclusive branches")
d = cohort.loc[cohort["cycle_start"].le(2007)]
branches = pd.Series(np.select(
    [
        d["prostate_any_positive"],
        ~d["prostate_any_positive"] & d["prostate_any_negative"],
        d["prostate_all_missing"] & d["RIDAGEYR"].lt(40),
        d["prostate_all_missing"] & d["RIDAGEYR"].ge(40),
    ],
    ["any prostate item positive", "questionnaire negative/no positive",
     "no questionnaire data, age <40", "no questionnaire data, age >=40"],
    default="other",
)).value_counts()
print(branches.to_string())

print("\n2013-2018 reason codes")
print(cohort.loc[cohort["cycle_start"].ge(2013), "reason_codes"].value_counts().to_string())

print("\nConcurrent medication counts, strict adult cohort")
adult = cohort["RIDAGEYR"].ge(18)
for cls in ["class_strict", "class_l989_age"]:
    for group, gm in [("retained", cohort[cls] & adult), ("excluded", ~cohort[cls] & adult)]:
        q = cohort.loc[gm]
        print(cls, group, len(q), int(q.bph_specific.sum()), int(q.ambiguous.sum()), int(q.any_broad.sum()))
