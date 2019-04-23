import numpy as np
import re

# a = np.array(['j0.jpg', 'sadf/5555.jpg'])
# r = re.compile('\d+')

r = r"\d+"
new_order = np.fromregex('hardPredictionPaths.txt', r, [('order', np.int16)])
predictions = np.genfromtxt('estimatedHardLabels.txt', delimiter=",", dtype='str')

print(new_order)
print(predictions)
# print(len(new_order))
# print(len(predictions))

arr_inds = new_order.argsort()

print(new_order[arr_inds])

new_predictions = predictions[arr_inds]

print(new_predictions)

# np.savetxt('correct_easy_labels.txt', new_predictions, fmt='%s', delimiter=',')
with open("correct_hard_labels.txt", "w") as file:
    file.write(str(new_predictions.tolist()))