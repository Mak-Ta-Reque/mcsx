import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches


data = np.empty((3,3, 10000))
df_55 = pd.read_csv('/mnt/sda/abka03-data/mcx/55_nonrobust/output_55_nonrobust.csv')
df_54 = pd.read_csv('/mnt/sda/abka03-data/mcx/54_nonrobust/output_54_nonrobust.csv')
df_76 = pd.read_csv('/mnt/sda/abka03-data/mcx/76_nonrobust/output_76_nonrobust.csv')

df_155 = pd.read_csv('/mnt/sda/abka03-data/mcx/155_nonrobust/output_155_nonrobust.csv')
df_154 = pd.read_csv('/mnt/sda/abka03-data/mcx/154_nonrobust/output_154_nonrobust.csv')
df_198 = pd.read_csv('/mnt/sda/abka03-data/mcx/198_nonrobust/output_198_nonrobust.csv')

# df_255 = pd.read_csv('/mnt/sda/abka03-data/mcx/255_nonrobust/output_255_nonrobust.csv')
# df_254 = pd.read_csv('/mnt/sda/abka03-data/mcx/254_nonrobust/output_254_nonrobust.csv')
# df_273 = pd.read_csv('/mnt/sda/abka03-data/mcx/273_nonrobust/output_273_nonrobust.csv')


data[0,0,:] = df_55['mse_diff_tri'].tolist()
data[0,1,:] = df_54['mse_diff_tri'].tolist()
data[0,2,:] = df_76['mse_diff_tri'].tolist()

data[1,0,:] = df_155['mse_diff_tri'].tolist()
data[1,1,:] = df_154['mse_diff_tri'].tolist()
data[1,2,:] = df_198['mse_diff_tri'].tolist()

# data[2,0,:] = df_255['mse_diff_tri'].tolist()
# data[2,1,:] = df_254['mse_diff_tri'].tolist()
# data[2,2,:] = df_273['mse_diff_tri'].tolist()




data_a = [df_55['mse_diff_tri'].tolist(), df_54['mse_diff_tri'].tolist(), df_76['mse_diff_tri'].tolist()]
data_b = [df_155['mse_diff_tri'].tolist(), df_154['mse_diff_tri'].tolist(), df_198['mse_diff_tri'].tolist()]

# Create labels for your lists
labels_a = ['df_55', 'df_54', 'df_76']
labels_b = ['df_155', 'df_154', 'df_198']
def setBoxColors(bp):
    colors = ['blue', 'red', 'green']
    for i in range(len(bp['boxes'])):
        plt.setp(bp['boxes'][i], color=colors[i])
        plt.setp(bp['caps'][2*i], color=colors[i])
        plt.setp(bp['caps'][2*i + 1], color=colors[i])
        plt.setp(bp['whiskers'][2*i], color=colors[i])
        plt.setp(bp['whiskers'][2*i + 1], color=colors[i])
        plt.setp(bp['fliers'][i], color=colors[i])
        plt.setp(bp['medians'][i], color=colors[i])

# Some fake data to plot
A= data_a
B = data_b
plt.rcParams.update({'font.size': 12})
fig = plt.figure()
ax = fig.add_subplot(111)

# first boxplot pair
bp = ax.boxplot(A, positions = [0.5, 1, 1.5], widths = 0.3)
setBoxColors(bp)

# second boxplot pair
bp = ax.boxplot(B, positions = [3.5, 4, 4.5], widths = 0.3)
setBoxColors(bp)

# set axes limits and labels
ax.set_xlim(0,5)
ax.set_ylim(0,0.6)
ax.set_xticks([1, 4])
ax.set_ylabel('MSE', fontsize = 12)
ax.set_xticklabels(['Simple Attack', 'Red Herring'], fontsize = 12)
# draw temporary red and blue lines and use them to create a legend
hB, = plt.plot([1,1],'b-')
hR, = plt.plot([1,1],'r-')
hL, = plt.plot([1,1],'g-')
legend = ax.legend((hB, hR, hL),('Grad', 'G-CAM', 'R-CAM'), loc='upper center')
hB.set_visible(False)
hR.set_visible(False)
hL.set_visible(False)
plt.tight_layout() # pad parameter adds space between the two plots
plt.show()


plt.savefig('boxcompare.pdf')