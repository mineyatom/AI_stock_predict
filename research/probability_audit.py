"""
V10.1 Probability Audit

使用 baseline_*_predictions.csv 產生 Calibration 報告。

用法：
python research/probability_audit.py research/results/baseline_2330_TW_predictions.csv
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np

def expected_calibration_error(df, bins):
    ece=0.0
    total=len(df)
    rows=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        part=df[(df.confidence>=lo)&(df.confidence<hi)]
        if len(part)==0:
            rows.append([f"{int(lo*100)}-{int(hi*100)}%",0,np.nan,np.nan,np.nan])
            continue
        acc=part.correct.mean()
        conf=part.confidence.mean()
        gap=conf-acc
        ece += abs(gap)*len(part)/total
        rows.append([f"{int(lo*100)}-{int(hi*100)}%",len(part),acc,conf,gap])
    table=pd.DataFrame(rows,columns=["Bin","Samples","ActualAccuracy","AverageConfidence","CalibrationGap"])
    return ece,table

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("csv")
    args=ap.parse_args()
    df=pd.read_csv(args.csv)
    bins=[0.50,0.55,0.60,0.65,0.70,0.80,0.90,1.000001]
    ece,table=expected_calibration_error(df,bins)
    valid=table.dropna()
    mce=valid.CalibrationGap.abs().max() if not valid.empty else np.nan
    print("="*60)
    print("Probability Audit")
    print("="*60)
    print(f"Samples : {len(df)}")
    print(f"Accuracy: {df.correct.mean():.2%}")
    print(f"Avg Conf: {df.confidence.mean():.2%}")
    print(f"ECE     : {ece:.4f}")
    print(f"MCE     : {mce:.4f}")
    print()
    t=table.copy()
    for c in ["ActualAccuracy","AverageConfidence","CalibrationGap"]:
        t[c]=t[c].apply(lambda x:"-" if pd.isna(x) else f"{x:.2%}")
    print(t.to_string(index=False))
    out=Path(args.csv).with_name(Path(args.csv).stem+"_probability_audit.csv")
    table.to_csv(out,index=False,encoding="utf-8-sig")
    print(f"\nSaved: {out}")

if __name__=="__main__":
    main()
