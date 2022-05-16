from textattack.datasets import HuggingFaceDataset
from textattack.augmentation import Augmenter
import pandas as pd
import os
# transformations
from textattack.transformations import WordSwapHowNet

# init
dataset = HuggingFaceDataset("ag_news", None, "test")

# trying all transformations
transformation = WordSwapHowNet()
augmenter = Augmenter(transformation=transformation)
list1 = []
print("Start WordSwapHowNet.")
for i in range(dataset.__len__()):
    print(i)
    aug1 = augmenter.augment(dataset[i][0]['text'])
    list1.append([aug1, dataset[i][1]])
df = pd.DataFrame(list1, columns=['text', 'label'])

# write to csv
os.system("touch wordswap_hownet.csv")
df.to_csv('wordswap_hownet.csv')
print("Done WordSwapHowNet.")