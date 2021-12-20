"""
Author: Ruidong Li (lrd19@mails.tsinghua.edu.cn), Ting Sun (ting.sun@reading.ac.uk)
performance_plot.py (c) 2021
Desc: generate metric plot; inspired by https://www.nature.com/articles/s41597-021-01079-3/figures/3
Created:  2021-12-11T09:31:26.685Z
"""

import os
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
from matplotlib.collections import PatchCollection
from matplotlib.patches import Wedge


def plot_wedge_quadrant(ax, dta, cmap_func, edgecolor="grey"):
    p = [Wedge(center=(0.5, 0.5), r=1, theta1=-45+90*s, theta2=45+90*s, 
                facecolor=cmap_func.to_rgba(v), edgecolor=edgecolor) for s, v in enumerate(dta)]
    
    p = PatchCollection(p, match_original=True)
    _ = ax.add_collection(p)
    _ = ax.xaxis.set_ticks([])
    _ = ax.yaxis.set_ticks([])


def model_wedge_plot(res_path):
    # ------read data
    with open(res_path, "r") as dta_tmp:
        res_dta = json.load(dta_tmp)
    
    # ------get the list for looping
    case_list = list(res_dta.keys())

    case_tmp = case_list[0]
    resolution_list = list(res_dta[case_tmp].keys())

    resolution_tmp = resolution_list[0]
    model_list = []
    for m in res_dta[case_tmp][resolution_tmp].keys():
        if m == "baggingSVR":
            m = "SVR"
        model_list.append(m)

    model_tmp = model_list[0]
    metric_list = list(res_dta[case_tmp][resolution_tmp][model_tmp].keys())

    # ------specify the plotting settings
    # ---------create the figure and axes
    n_row = len(case_list)
    num_model = len(model_list)
    num_metric = len(metric_list)
    n_col = num_model * num_metric

    size_base = 1.0
    fig, axes = plt.subplots(
        n_row,
        n_col,
        figsize=(n_col * 1 * size_base, n_row * 0.6 * size_base),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )

    # ------specify the color mapping and text label for metrics
    cmap_ref = {
        "R^2": "Reds",
        "nME^2": "RdBu_r",
        "nRMSE_centered^2": "Purples",
        "NMAD": "Blues"
    }

    range_ref = {
        "R^2": [0.0, 1.0],
        "nME^2": [0.0, 1.0],
        "nRMSE_centered^2": [0.0, 1.0],
        "NMAD": [0.0, 1.0]
    }

    cmap_norm_ref = {}
    for metric in cmap_ref.keys():
        v_min, v_max = range_ref[metric]
        cmap_norm_ref[metric] = mpc.Normalize(vmin=v_min, vmax=v_max)
    
    cbar_label_ref = {
        "R^2": "$R^2$",
        "nME^2": "$\mathrm{nME}^2$",
        "nRMSE_centered^2": "${\mathrm{nRMSE}_{\mathrm{centered}}}^2$",
        "NMAD": "NMAD"
    }

    outskirt_label_ref = {
        "R^2": "$R^2$",
        "nME^2": "$\mathrm{nME}^2$",
        "nRMSE_centered^2": "${\mathrm{nRMSE}_{\mathrm{centered}}}^2$",
        "NMAD": "NMAD"
    }

    # ------plot the wedge quadrant
    for row_id in range(0, n_row):
        case_tmp = case_list[row_id]
        for metric_id in range(0, num_metric):
            col_id_shift = metric_id * num_model
            metric_tmp = metric_list[metric_id]

            norm = cmap_norm_ref[metric_tmp]
            cmap = mpl.cm.get_cmap(cmap_ref[metric])
            cmap_func = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

            for model_id in range(0, num_model):
                col_id = col_id_shift + model_id
                model_tmp = model_list[model_id]
                if model_tmp == "SVR":
                    case_res_dta = [res_dta[case_tmp][res]["baggingSVR"][metric_tmp] for res in ["100m", "250m"]]
                    case_res_dta = case_res_dta + [res_dta[case_tmp][res]["SVR"][metric_tmp] for res in ["500m", "1000m"]]
                else:
                    case_res_dta = [res_dta[case_tmp][res][model_tmp][metric_tmp] for res in resolution_list]
                    
                ax = axes[row_id, col_id]
                plot_wedge_quadrant(ax=ax, dta=case_res_dta, cmap_func=cmap_func)
                print(case_tmp, model_tmp, metric_tmp)

                if row_id == 0:
                    _ = ax.set_xlabel(model_tmp)
                    _ = ax.xaxis.set_label_position("top")
                
                if col_id == 0:
                    _ = ax.set_ylabel(case_tmp)
    
    # ------add the colorbar
    list_cax = []
    for c, metric in enumerate(metric_list[-1::-1]):
        norm = cmap_norm_ref[metric]
        cmap = mpl.cm.get_cmap(cmap_ref[metric])
        cmap_func = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(
            cmap_func,
            ax=axes,
            orientation="horizontal",
            fraction=0.04,
            panchor=(0.1, 0.0),
            anchor=(1, 1),
            shrink=0.7,
            pad=0.01,
        )

        cax = cbar.ax
        lbl = metric + "\n" + cbar_label_ref[metric]
        _ = cax.text(
            -0.08,
            0.0,
            lbl,
            size=11,
            va="center",
            ha="center",
            transform=cax.transAxes,
        )
        list_cax.append(cax)
    
    # ------add the outskirt labels of metrics
    for c0, metric in enumerate(metric_list):
        x = []
        for c1, model in enumerate(model_list):
            c = c0 * len(model_list) + c1
            ax_tmp = axes[0, c]
            bbox = ax_tmp.get_position()
            x_ax, _, w, _ = bbox.bounds
            x.append(x_ax + w / 2)
        x = np.mean(x)

        lbl = metric
        _ = fig.text(
            x,
            1,
            lbl,
            rotation="horizontal",
            ha="center",
            va="bottom",
            fontsize=13,
        )

    plt.savefig("wedge_preformance.pdf", bbox_inches='tight')



if __name__ == "__main__":
    res_path_ref = {
        "$H_{\mathrm{ave}}$": "./height.json",
        "$\lambda_p$": "./footprint.json"
    }
    # model_wedge_plot(res_path=res_path_ref["$\lambda_p$"])

    with open("./height.json", "r") as dta_tmp:
        res_dta = json.load(dta_tmp)

    num = 0
    for k in res_dta.keys():
        num_tmp = res_dta[k]["100m"]["rf"]["N"]
        print(k, num_tmp)
        num += num_tmp
    print(num)


