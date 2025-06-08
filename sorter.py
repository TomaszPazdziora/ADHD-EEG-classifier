import pandas as pd

df = pd.read_csv("features.csv")
# zamień "wartość" na nazwę swojej kolumny
df_sorted = df.sort_values(by="Różnica między średnimi")
df_sorted.to_csv("posortowane.csv", index=False)
