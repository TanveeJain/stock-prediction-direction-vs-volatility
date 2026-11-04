import glob
import os
import pandas as pd

accs, aucs = [], []
for f in sorted(glob.glob("results/benchmark_*.csv")):
    name = os.path.basename(f)[len("benchmark_"):-len(".csv")]
    t = pd.read_csv(f).set_index("model")
    accs.append(t["acc"].rename(name))
    aucs.append(t["auc"].rename(name))

if not accs:
    raise SystemExit("no benchmark csv files in results/")

acc = pd.concat(accs, axis=1)
auc = pd.concat(aucs, axis=1).drop(index="baseline", errors="ignore")
acc = acc.sort_values(acc.columns[0], ascending=False)
if "baseline" in acc.index:
    acc = pd.concat([acc.drop(index="baseline"), acc.loc[["baseline"]]])


def md(t):
    head = "| model | " + " | ".join(t.columns) + " |"
    line = "|---|" + "---|" * len(t.columns)
    body = ["| " + str(i) + " | " + " | ".join("" if pd.isna(v) else f"{v:.3f}" for v in r) + " |" for i, r in t.iterrows()]
    return "\n".join([head, line] + body)


text = "# Accuracy\n\n" + md(acc) + "\n\n# AUC (0.5 = no skill)\n\n" + md(auc) + "\n"
open("results/summary.md", "w").write(text)
print(text)
