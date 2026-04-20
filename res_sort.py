import pandas as pd
import glob

dfs = []

for file in glob.glob("res/narrativeqa/*.jsonl"):

    df = pd.read_json(file, lines=True)

    name=file.split("/")[-1].replace(".jsonl","")
    parts=name.split("-")

    if len(parts)==2:
        kv=parts[0]
        quant="none"
        context=parts[1]
    else:
        kv=parts[0]
        quant=parts[1]
        context=parts[-1]

    df["kv"]=kv
    df["quant"]=quant
    df["context"]=int(context)

    dfs.append(df)

all_df=pd.concat(dfs)

all_df.to_csv("all_results.csv",index=False)