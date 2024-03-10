import pandas as pd

df=pd.read_csv('data/airports.csv', header='infer')

print(df.tail(12)["iso_country"])
print()

print(df.loc(1))
print(df.iloc[1])
print()

print(df[df['iso_country'] == 'Poland'])
print()

print(df[df.apply(lambda row: str(row['municipality']) not in str(row['name']), axis=1)])
print()

df['elevation'] = df['elevation'] * 0.3048
print(df)
print()

print(df[df['iso_country'].map(df['iso_country'].value_counts()) == 1])
