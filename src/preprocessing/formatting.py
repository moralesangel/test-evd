import pandas as pd

POSITIVE_LABEL = "/m/03j1ly"
SET = 'dataset/large_set.csv'
NEW_SET = 'dataset/large_new.csv'
ROWS = None

CSV = pd.read_csv(SET,
                    sep=', ',
                    engine='python',
                    comment='#',
                    nrows=ROWS)

labels_field = CSV[CSV.columns[-1]]
new_column = []

for row in labels_field:
    index = row.find(POSITIVE_LABEL)
    if index == -1:
        new_column.append(0)
    else:
        new_column.append(1)

CSV['output'] = new_column

CSV = CSV.map(lambda x: x.replace('"', '') if isinstance(x, str) else x)

CSV.to_csv(NEW_SET, index=False)
