import numpy as np
import pandas as pd


df = pd.read_csv('https://github.com/Ulvi-Movs/titanic/raw/main/train.csv')

df = df.drop(columns=["PassengerId", "Name", "Ticket"])

df['Has_Cabin'] = df['Cabin'].notnull().astype(int)
