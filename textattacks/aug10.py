from textattack.datasets import HuggingFaceDataset
from textattack.augmentation import Augmenter
import pandas as pd
import os
import nltk
# transformations
from textattack.transformations import WordSwapWordNet

# init
nltk.download('omw-1.4')
dataset = HuggingFaceDataset("ag_news", None, "test")

# trying all transformations
transformation = WordSwapWordNet()
augmenter = Augmenter(transformation=transformation)
list1 = []
print("Start WordSwapWordNet.")
for i in range(dataset.__len__()):
    print(i)
    aug1 = augmenter.augment(dataset[i][0]['text'])
    list1.append([aug1, dataset[i][1]])
df = pd.DataFrame(list1, columns=['text', 'label'])

# write to csv
os.system("touch wordswap_wordnet.csv")
df.to_csv('wordswap_wordnet.csv')
print("Done WordSwapWordNet.")