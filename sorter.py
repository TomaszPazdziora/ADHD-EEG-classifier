import pandas as pd
import csv
from collections import defaultdict

if __name__ == "__main__":
    mode = input("Would you like to 'sort' or 'count' sorted features:")
    if mode == 'sort':
        df = pd.read_csv("features.csv")
        df_sorted = df.sort_values(by="Różnica między średnimi")
        df_sorted.to_csv("posortowane.csv", index=False)
    elif mode == 'count':
        f4_data = defaultdict(int)
        other_data = defaultdict(int)

        with open('posortowane_cechy.csv', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for i, row in enumerate(reader):
                desc = row[0]
                parts = desc.split(", ")
                parsed = {p.split(": ")[0]: p.split(": ")[1] for p in parts}
                key = (parsed["fala"], parsed["cecha"])
                electrode = parsed["elektroda"]

                if electrode == 'F4':
                    f4_data[key] += i
                else:
                    other_data[key] += i

        # Zapis F4.csv
        with open('z_F4.csv', mode='w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(['fala', 'cecha', 'suma_indeksów'])
            for (fala, cecha), score in sorted(f4_data.items(), key=lambda x: x[1], reverse=True):
                writer.writerow([fala, cecha, score])

        # Zapis inne.csv
        with open('z_inne.csv', mode='w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(['fala', 'cecha', 'suma_indeksów'])
            for (fala, cecha), score in sorted(other_data.items(), key=lambda x: x[1], reverse=True):
                writer.writerow([fala, cecha, score])

    else:
        print("unknown option")
