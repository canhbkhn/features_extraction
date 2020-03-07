import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calc_woe_and_iv(pdf_data, attr_name, n_bins=10, wanna_plot=False):
    # find bounded range
    min_x = pdf_data[attr_name].min()
    max_x = pdf_data[attr_name].max()
    print("Min: {}, Max: {}".format(min_x, max_x))
    
    # extracting histogram    
    hist_all = np.histogram(pdf_data[attr_name], range=(min_x, max_x), bins=n_bins)
    hist_good = np.histogram(pdf_data.query("TARGET == 0")[attr_name], range=(min_x, max_x), bins=n_bins)
    hist_bad = np.histogram(pdf_data.query("TARGET == 1")[attr_name], range=(min_x, max_x), bins=n_bins)
    
    # get total number of good and bad for normalization
    total_good = hist_good[0].sum()
    total_bad = hist_bad[0].sum()
    print("Num good: {}, Num bad: {}".format(total_good, total_bad))
    
    # convert histogram to series
    s_all = pd.Series(dict(zip(hist_all[1], hist_all[0])))
    s_all.rename(lambda x: "{0:.2f}".format(x), inplace=True)
    
    s_good = pd.Series(dict(zip(hist_good[1], hist_good[0])))
    s_good.rename(lambda x: "{0:.2f}".format(x), inplace=True)
    
    s_bad = pd.Series(dict(zip(hist_bad[1], hist_bad[0])))
    s_bad.rename(lambda x: "{0:.2f}".format(x), inplace=True)
    
    # calculate distribution of good/bad
    distr_good = s_good / total_good
    distr_bad = s_bad / total_bad
    
    # getting denominator indices not equal zeros
    idx_filtered = distr_bad[(distr_bad != 0) & (np.isfinite(distr_bad) & (distr_bad > 1e-04))].index
    
    # calculate woe
    woe = np.log(distr_good[idx_filtered]) - np.log(distr_bad[idx_filtered])
    
    # calculate iv
    iv = ((distr_good[idx_filtered] - distr_bad[idx_filtered]) * woe).sum()
    
    # plot (optional)
    if wanna_plot:
        ax01 = s_all[idx_filtered].plot(kind="bar", colormap="Paired")
        ax01.set_xlabel(attr_name)

        # 
        ax02 = ax01.twinx()  # instantiate a second axes that shares the same x-axis
        woe.to_frame("WoE").plot(kind="line", ax=ax02, legend=True)

        # 
        plt.title("IV = {0:.5f}".format(iv))
        plt.legend(loc="best")
        plt.show()
    
    return woe, iv