from textattack.datasets import HuggingFaceDataset
from textattack.augmentation import Augmenter
import pandas as pd
import os
# transformations
from textattack.transformations import WordSwapNeighboringCharacterSwap

# init
dataset = HuggingFaceDataset("ag_news", None, "test")

# trying all transformations
transformation = WordSwapNeighboringCharacterSwap()
augmenter = Augmenter(transformation=transformation)
list1 = []
print("Start WordSwapNeighboringCharacterSwap.")
for i in range(dataset.__len__()):
    print(i)
    aug1 = augmenter.augment(dataset[i][0]['text'])
    list1.append([aug1, dataset[i][1]])
df = pd.DataFrame(list1, columns=['text', 'label'])

# write to csv
os.system("touch wordswap_neighbor_char.csv")
df.to_csv('wordswap_neighbor_char.csv')
print("Done WordSwapNeighboringCharacterSwap.")