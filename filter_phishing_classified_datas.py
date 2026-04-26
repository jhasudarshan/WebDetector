from operator import itemgetter

import pandas as pd

# Load the CSV file
file_path = './Helper/ResultDataSet/last_result.csv'
df = pd.read_csv(file_path)

results = []
new_urls_file = './Helper/NewDataSet/dataset_2.csv'
for i, row in df.iterrows():
    if row["prediction"] != "benign":
        record = {
            "url": row["url"],
            "label" : row["prediction"],
        }
        results.append(record)


pd.DataFrame(results).to_csv(new_urls_file, index=False)
print("Size: ", len(results))