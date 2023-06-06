#%%
import matplotlib.pyplot as plt
import pandas as pd


fig, ax = plt.subplots()
fig.set_figheight(5)
fig.set_figwidth(20)
fig.subplots_adjust(right=0.75)

twin1 = ax.twinx()
twin2 = ax.twinx()
twin3 = ax.twinx()

# Offset the right spine of twin2.  The ticks and label have already been
# placed on the right by twinx above.
twin2.spines.right.set_position(("axes", 1.2))
twin3.spines.right.set_position(("axes", 1.4))

x=[] #checkpoint numbers
# Different loss values
y1 = []
y2 = []
y3 = []

# An example bert-base-uncased-goemo-vae-b16-e50-with-kl-0828-2015-0
# x = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000]
# y1 = [14.746339,12.598419,11.307368,10.573695,9.861722,9.214516,8.655242,8.156064,7.68275,7.348649,6.989104,6.749061,6.506948,6.353395,6.191617,6.051216,5.928932,5.844477,5.791457,5.711079,5.671912,5.611872,5.57564,5.542349,5.498521,5.494482,5.488085,5.468708,5.45833,5.456627,5.442178,5.440772,5.421768,5.410783,5.406324,5.406286,5.381073,5.375143,5.387301,5.378551,5.361524,5.380447,5.380738]
# y2 = [0.126079,0.093459,0.075389,0.065622,0.056869,0.049435,0.043553,0.038491,0.034036,0.031017,0.027978,0.026024,0.024144,0.022953,0.021783,0.020768,0.019897,0.019316,0.018924,0.018402,0.01812,0.017751,0.017492,0.017271,0.016993,0.016956,0.016905,0.016765,0.016695,0.016655,0.016576,0.016551,0.016438,0.016384,0.016337,0.016329,0.016181,0.016148,0.016202,0.016153,0.016053,0.016151,0.016152]
# y3 = [1.856451,0.006161,0.009639,0.028937,0.155398,0.28805,0.423585,0.525414,0.625398,0.65344,0.697049,0.685687,0.694344,0.663642,0.647515,0.621636,0.591096,0.56368,0.526426,0.522964,0.4917,0.480798,0.459367,0.435599,0.429184,0.40891,0.380349,0.371238,0.359253,0.350782,0.331573,0.321783,0.323993,0.320215,0.310913,0.294807,0.304816,0.299528,0.289887,0.284532,0.291957,0.282411,0.27738]

loss_info = pd.read_csv("or you can save them as a csv file, blabla.csv", sep=",", header=0)
y1, y2, y3 = loss_info['pw'], loss_info['kl'], loss_info['or'] 
x = range(1, len(y1)+1)
y1_top = max(y1)
y2_top = max(y2)
y3_top = max(y3)
y1_bot = min(y1)
y2_bot = min(y2)
y3_bot = min(y3)

p1, = twin1.plot(x, y1, "b-", label="pairwise")
p2, = twin2.plot(x, y2, "r-", label="kl")
p3, = twin3.plot(x, y3, "g-", label="orthogonal")

ax.set_xlim(0, len(y1)+1)
twin1.set_ylim(y1_bot,y1_top)
twin2.set_ylim(y2_bot, y2_top)
twin3.set_ylim(y3_bot, y3_top)

ax.set_xlabel("Training time")
twin1.set_ylabel("Pairwise")
twin2.set_ylabel("KL")
twin3.set_ylabel("Orthogonal")

twin1.yaxis.label.set_color(p1.get_color())
twin2.yaxis.label.set_color(p2.get_color())
twin3.yaxis.label.set_color(p3.get_color())

tkw = dict(size=4, width=1.5)
twin1.tick_params(axis='y', colors=p1.get_color(), **tkw)
twin2.tick_params(axis='y', colors=p2.get_color(), **tkw)
twin3.tick_params(axis='y', colors=p3.get_color(), **tkw)
ax.tick_params(axis='x', **tkw)

ax.legend(handles=[p1, p2, p3])

plt.show()
# %%
