import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

# Load the dataset
df = pd.read_csv('C:/Workspaces/ml-eta/ride-eta-prediction-team11/NYC.csv')

# Display the first 5 rows of the DataFrame
display(df.head())

# Display basic information about the DataFrame
df.info()

# Display descriptive statistics
display(df.describe(include='all'))

