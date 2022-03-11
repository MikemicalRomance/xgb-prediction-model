import pandas as pd
pd.set_option("display.max_rows", None)

##Fetch data 
pets_at_home_df = pd.read_csv("gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv") 
print(pets_at_home_df.head(5))
## initial EDA; data looks pretty clean all 14 columns have 11537 non null values 
import pdb; pdb.set_trace()