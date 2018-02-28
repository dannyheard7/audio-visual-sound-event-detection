import dill as pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math

predictions = pickle.load(open( "predictions.pkl", "rb" ))

num_classes = len(predictions)
cols = 2
rows = math.ceil(num_classes / cols)
fig, axarr = plt.subplots(rows, cols, figsize=(185, 40))

# Should sort ImageNet classes so they are in the same order for each dcase class and have a key

dcase_labels = list(predictions.keys())
count = 0
for i in range(0, rows):
    for j in range(0, cols):
        if len(dcase_labels) == count:
            break

        class_name = dcase_labels[count]
        class_predictions = predictions[class_name]
        dict_len = len(class_predictions)

        axarr[i, j].bar(range(dict_len), list(class_predictions.values()), align='center')
        axarr[i, j].set_title("ImageNet Predicitions for {}".format(class_name))
        axarr[i, j].set_xticks(range(dict_len))
        axarr[i, j].set_xticklabels(list(class_predictions.keys()), rotation='vertical')  
        count +=1 

fig.tight_layout()
plt.savefig('predictions-bar-chart.png')