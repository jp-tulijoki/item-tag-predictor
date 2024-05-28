import pandas as pd

RESULT_FILE = ""

result_df = pd.read_csv(RESULT_FILE)
result_df["overall_mae"] = result_df["overall_mae"].apply(lambda x: round(x, 4))
result_df["ci +/-"] = result_df["ci +/-"].apply(lambda x: round(x, 4))
result_df["tagdl_diff"] = result_df["tagdl_diff"].apply(lambda x: round(x * 100, 2))
result_df["p-value"] = result_df["p-value"].apply(lambda x: round(x, 4))
df_tolatex = result_df.to_latex(index=True)
print(df_tolatex)