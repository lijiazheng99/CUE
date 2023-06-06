import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rc_file_defaults()

def visual_deltae(dataframe, name):
    ax1 = sns.set_style(style=None, rc=None)    

    fig, ax1 = plt.subplots(figsize=(7,4))  

    latent_sizes = ['0-10','10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
    accs = dataframe['acc'] 
    f1s = dataframe['f1']
    entro = dataframe['mean_entro']
    eces = dataframe['ece']

    accs, f1s, entro, eces = accs[::-1], f1s[::-1], entro[::-1], eces[::-1] 


    ax = sns.barplot(x=latent_sizes, y=eces, alpha=0.5, palette="Blues_d", ax=ax1, label="ECE") 

    twin1 = ax.twinx()
    twin2 = ax.twinx()
    twin3 = ax.twinx()
    twin1.spines.right.set_position(("axes", 1.0))
    twin2.spines.right.set_position(("axes", 1.15))
    twin3.spines.right.set_position(("axes", 1.3))  

    y3, y2, y1  = accs, f1s, entro
    x = latent_sizes
    y1_top = max(y1)
    y2_top = max(y2)
    y3_top = max(y3)
    y1_bot = min(y1)
    y2_bot = min(y2)
    y3_bot = min(y3)    

    p1, = twin1.plot(x, y1, "b-", label=r'$\|\|\Delta e\|\|_1$')
    p2, = twin2.plot(x, y2, "r-", alpha=0.5, label="Macro F1")
    p3, = twin3.plot(x, y3, "g-", alpha=0.5, label="Accuracy")  

    # Adjust if need
    # ax.set_ylim(min(eces) - (max(eces)-min(eces))*0.1, max(eces) + (max(eces)-min(eces))*0.1 )
    # twin1.set_ylim(y1_bot,y1_top)
    # twin2.set_ylim(0.20,0.70)
    # twin3.set_ylim(0.45,0.65) 

    ax.set_xlabel("Ranked Latent Dimensions")
    ax.set_ylabel(r'ECE')
    twin1.set_ylabel(r'mean $H$')
    twin2.set_ylabel("F1")
    twin3.set_ylabel("Accuracy")    

    twin1.yaxis.label.set_color(p1.get_color())
    twin2.yaxis.label.set_color(p2.get_color())
    twin3.yaxis.label.set_color(p3.get_color()) 

    tkw = dict(size=4, width=1.5)
    twin1.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin2.tick_params(axis='y', colors=p2.get_color(), **tkw)
    twin3.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw) 

    ax.legend(handles=[p1, p2, p3])
    ax.figure.savefig(f"{name}.png", dpi=300)
    plt.show()