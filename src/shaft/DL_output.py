import os
import re
import imageio
import numpy as np
import matplotlib.pyplot as plt

from visualization import *


def extract_record(output_file, num_quantile=22):
    f = open(output_file, "r")
    lines = f.readlines()
    res_dict = {"Training": {"Loss": [],
                             "R^2": [],
                             "RMSE": [],
                             "MAE": [],
                             "Quantiles": [],
                             "confusion_matrix": []},
                "Validation": {"Loss": [],
                               "R^2": [],
                               "RMSE": [],
                               "MAE": [],
                               "Quantiles": [],
                               "confusion_matrix": []}}

    idx = 0
    for l in lines:
        if "Training Loss" in l:
            loss = float(re.findall(r"(\d+.\d+)\n", l)[0])
            res_dict["Training"]["Loss"].append(loss)
        elif "Training R^2" in l:
            r2 = float(re.findall(r"(\d+.\d+)\n", l)[0])
            res_dict["Training"]["R^2"].append(r2)
        elif "Training RMSE" in l:
            rmse = float(re.findall(r"(\d+.\d+)\n", l)[0])
            res_dict["Training"]["RMSE"].append(rmse)
        elif "Training MAE" in l:
            mae = float(re.findall(r"(\d+.\d+)\n", l)[0])
            res_dict["Training"]["MAE"].append(mae)
        elif "Training Quantiles" in l:
            src_line = lines[idx + 1]
            src_arr = np.fromstring(src_line, sep=" ")
            res_dict["Training"]["Quantiles"].append(np.expand_dims(src_arr, axis=-1))
        elif "Training confusion matrix" in l:
            src_line = lines[idx + 1:idx + 1 + num_quantile]
            src_arr = np.array([np.fromstring(src, sep=" ") for src in src_line])
            res_dict["Training"]["confusion_matrix"].append(np.expand_dims(src_arr, axis=-1))
        elif "Validation Loss" in l:
            loss = float(re.findall(r"(\d+.\d+)\n", l)[0])
            res_dict["Validation"]["Loss"].append(loss)
        elif "Validation R^2" in l:
            r2 = float(re.findall(r"(\d+.\d+)\n", l)[0])
            res_dict["Validation"]["R^2"].append(r2)
        elif "Validation RMSE" in l:
            rmse = float(re.findall(r"(\d+.\d+)\n", l)[0])
            res_dict["Validation"]["RMSE"].append(rmse)
        elif "Validation MAE" in l:
            mae = float(re.findall(r"(\d+.\d+)\n", l)[0])
            res_dict["Validation"]["MAE"].append(mae)
        elif "Validation Quantiles" in l:
            src_line = lines[idx + 1]
            src_arr = np.fromstring(src_line, sep=" ")
            res_dict["Validation"]["Quantiles"].append(np.expand_dims(src_arr, axis=-1))
        elif "Validation confusion matrix" in l:
            src_line = lines[idx + 1:idx + 1 + num_quantile]
            src_arr = np.array([np.fromstring(src, sep=" ") for src in src_line])
            res_dict["Validation"]["confusion_matrix"].append(np.expand_dims(src_arr, axis=-1))

        idx += 1

    res_dict["Training"]["Quantiles"] = np.concatenate(res_dict["Training"]["Quantiles"], axis=-1)
    res_dict["Validation"]["Quantiles"] = np.concatenate(res_dict["Validation"]["Quantiles"], axis=-1)
    res_dict["Training"]["confusion_matrix"] = np.concatenate(res_dict["Training"]["confusion_matrix"], axis=-1)
    res_dict["Validation"]["confusion_matrix"] = np.concatenate(res_dict["Validation"]["confusion_matrix"], axis=-1)

    return res_dict


def extract_record_MTL(output_file, num_quantile=22):
    f = open(output_file, "r")
    lines = f.readlines()
    res_dict = {"BuildingFootprint": {
        "Training": {"Loss": [], "R^2": [], "RMSE": [], "MAE": [], "Quantiles": [], "confusion_matrix": []},
        "Validation": {"Loss": [], "R^2": [], "RMSE": [], "MAE": [], "Quantiles": [], "confusion_matrix": []}
    },
        "BuildingHeight": {
        "Training": {"Loss": [], "R^2": [], "RMSE": [], "MAE": [], "Quantiles": [], "confusion_matrix": []},
        "Validation": {"Loss": [], "R^2": [], "RMSE": [], "MAE": [], "Quantiles": [], "confusion_matrix": []}
    }}

    idx = 0
    for l in lines:
        if "BuildingHeight" in l:
            if "Training Loss" in l:
                loss = float(re.findall(r"(\d+.\d+)\n", l)[0])
                res_dict["BuildingHeight"]["Training"]["Loss"].append(loss)
            elif "Training R^2" in l:
                r2 = float(re.findall(r"(\d+.\d+)\n", l)[0])
                res_dict["BuildingHeight"]["Training"]["R^2"].append(r2)
            elif "Training RMSE" in l:
                rmse = float(re.findall(r"(\d+.\d+)\n", l)[0])
                res_dict["BuildingHeight"]["Training"]["RMSE"].append(rmse)
            elif "Training MAE" in l:
                mae = float(re.findall(r"(\d+.\d+)\n", l)[0])
                res_dict["BuildingHeight"]["Training"]["MAE"].append(mae)
            elif "Training Quantiles" in l:
                src_line = lines[idx + 1]
                src_arr = np.fromstring(src_line, sep=" ")
                res_dict["BuildingHeight"]["Training"]["Quantiles"].append(np.expand_dims(src_arr, axis=-1))
            elif "Training confusion matrix" in l:
                src_line = lines[idx + 1:idx + 1 + num_quantile]
                src_arr = np.array([np.fromstring(src, sep=" ") for src in src_line])
                res_dict["BuildingHeight"]["Training"]["confusion_matrix"].append(np.expand_dims(src_arr, axis=-1))
            elif "Validation Loss" in l:
                loss = float(re.findall(r"(\d+.\d+)\n", l)[0])
                res_dict["BuildingHeight"]["Validation"]["Loss"].append(loss)
            elif "Validation R^2" in l:
                r2 = float(re.findall(r"(\d+.\d+)\n", l)[0])
                res_dict["BuildingHeight"]["Validation"]["R^2"].append(r2)
            elif "Validation RMSE" in l:
                rmse = float(re.findall(r"(\d+.\d+)\n", l)[0])
                res_dict["BuildingHeight"]["Validation"]["RMSE"].append(rmse)
            elif "Validation MAE" in l:
                mae = float(re.findall(r"(\d+.\d+)\n", l)[0])
                res_dict["BuildingHeight"]["Validation"]["MAE"].append(mae)
            elif "Validation Quantiles" in l:
                src_line = lines[idx + 1]
                src_arr = np.fromstring(src_line, sep=" ")
                res_dict["BuildingHeight"]["Validation"]["Quantiles"].append(np.expand_dims(src_arr, axis=-1))
            elif "Validation confusion matrix" in l:
                src_line = lines[idx + 1:idx + 1 + num_quantile]
                src_arr = np.array([np.fromstring(src, sep=" ") for src in src_line])
                res_dict["BuildingHeight"]["Validation"]["confusion_matrix"].append(np.expand_dims(src_arr, axis=-1))
        elif "BuildingFootprint" in l:
            if "Training Loss" in l:
                loss = float(re.findall(r"(\d+.\d+)\n", l)[0])
                res_dict["BuildingFootprint"]["Training"]["Loss"].append(loss)
            elif "Training R^2" in l:
                r2 = float(re.findall(r"(\d+.\d+)\n", l)[0])
                res_dict["BuildingFootprint"]["Training"]["R^2"].append(r2)
            elif "Training RMSE" in l:
                rmse = float(re.findall(r"(\d+.\d+)\n", l)[0])
                res_dict["BuildingFootprint"]["Training"]["RMSE"].append(rmse)
            elif "Training MAE" in l:
                mae = float(re.findall(r"(\d+.\d+)\n", l)[0])
                res_dict["BuildingFootprint"]["Training"]["MAE"].append(mae)
            elif "Training Quantiles" in l:
                src_line = lines[idx + 1]
                src_arr = np.fromstring(src_line, sep=" ")
                res_dict["BuildingFootprint"]["Training"]["Quantiles"].append(np.expand_dims(src_arr, axis=-1))
            elif "Training confusion matrix" in l:
                src_line = lines[idx + 1:idx + 1 + num_quantile]
                src_arr = np.array([np.fromstring(src, sep=" ") for src in src_line])
                res_dict["BuildingFootprint"]["Training"]["confusion_matrix"].append(np.expand_dims(src_arr, axis=-1))
            elif "Validation Loss" in l:
                loss = float(re.findall(r"(\d+.\d+)\n", l)[0])
                res_dict["BuildingFootprint"]["Validation"]["Loss"].append(loss)
            elif "Validation R^2" in l:
                r2 = float(re.findall(r"(\d+.\d+)\n", l)[0])
                res_dict["BuildingFootprint"]["Validation"]["R^2"].append(r2)
            elif "Validation RMSE" in l:
                rmse = float(re.findall(r"(\d+.\d+)\n", l)[0])
                res_dict["BuildingFootprint"]["Validation"]["RMSE"].append(rmse)
            elif "Validation MAE" in l:
                mae = float(re.findall(r"(\d+.\d+)\n", l)[0])
                res_dict["BuildingFootprint"]["Validation"]["MAE"].append(mae)
            elif "Validation Quantiles" in l:
                src_line = lines[idx + 1]
                src_arr = np.fromstring(src_line, sep=" ")
                res_dict["BuildingFootprint"]["Validation"]["Quantiles"].append(np.expand_dims(src_arr, axis=-1))
            elif "Validation confusion matrix" in l:
                src_line = lines[idx + 1:idx + 1 + num_quantile]
                src_arr = np.array([np.fromstring(src, sep=" ") for src in src_line])
                res_dict["BuildingFootprint"]["Validation"]["confusion_matrix"].append(np.expand_dims(src_arr, axis=-1))

        idx += 1

    res_dict["BuildingHeight"]["Training"]["Quantiles"] = np.concatenate(res_dict["BuildingHeight"]["Training"]["Quantiles"], axis=-1)
    res_dict["BuildingHeight"]["Validation"]["Quantiles"] = np.concatenate(res_dict["BuildingHeight"]["Validation"]["Quantiles"], axis=-1)
    res_dict["BuildingHeight"]["Training"]["confusion_matrix"] = np.concatenate(res_dict["BuildingHeight"]["Training"]["confusion_matrix"], axis=-1)
    res_dict["BuildingHeight"]["Validation"]["confusion_matrix"] = np.concatenate(res_dict["BuildingHeight"]["Validation"]["confusion_matrix"], axis=-1)

    res_dict["BuildingFootprint"]["Training"]["Quantiles"] = np.concatenate(res_dict["BuildingFootprint"]["Training"]["Quantiles"], axis=-1)
    res_dict["BuildingFootprint"]["Validation"]["Quantiles"] = np.concatenate(res_dict["BuildingFootprint"]["Validation"]["Quantiles"], axis=-1)
    res_dict["BuildingFootprint"]["Training"]["confusion_matrix"] = np.concatenate(res_dict["BuildingFootprint"]["Training"]["confusion_matrix"], axis=-1)
    res_dict["BuildingFootprint"]["Validation"]["confusion_matrix"] = np.concatenate(res_dict["BuildingFootprint"]["Validation"]["confusion_matrix"], axis=-1)

    return res_dict


def makeGif(res_dir):
    images = [f for f in os.listdir(res_dir) if (f.endswith(".png") and f.startswith("senet"))]
    images = sorted(images, key=lambda x: int(re.findall(r"_(\d+).png", x)[0]))

    print(images)

    with imageio.get_writer("res_senet.gif", mode="I", duration=0.3) as writer:
        for i in images:
            im = imageio.imread(os.path.join(res_dir, i))
            writer.append_data(im)


if __name__ == "__main__":
    # ---Single-Task Training
    '''
    senet_H_huber_file = os.path.join("DL_run", "res_file", "check_pt_senet_100m", "experiment_5", "out_senet18.txt")
    senet_H_huber_res = extract_record(senet_H_huber_file)

    metric = "MAE"

    senet_H_huber_training_loss = senet_H_huber_res["Training"][metric]
    senet_H_huber_validation_loss = senet_H_huber_res["Validation"][metric]

    n = 38
    num_epoch = np.arange(1, n + 1).astype(np.int)

    fig, ax = plt.subplots(1, 2, figsize=(20, 9))

    ax[0].plot(num_epoch, senet_H_huber_training_loss[0:n], color="royalblue", label="H, AdaptiveHuber, SENet18")
    ax[0].set_title("Training {0}".format(metric))
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel(metric)
    ax[0].legend()

    ax[1].plot(num_epoch, senet_H_huber_validation_loss[0:n], color="royalblue", label="H, AdaptiveHuber, SENet18")
    ax[1].set_title("Validation {0}".format(metric))
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel(metric)
    ax[1].legend()

    # plt.show()
    plt.savefig("train_valid_SENet_MAE.png", dpi=400)
    '''

    '''
    resnet_lnH_mse_file = os.path.join("DL_run", "res_file", "check_pt_resnet_100m", "experiment_4", "out_resnet18.txt")
    resnet_lnH_mse_res = extract_record(resnet_lnH_mse_file)
    senet_lnH_mse_file = os.path.join("DL_run", "res_file", "check_pt_senet_100m", "experiment_3", "out_senet18.txt")
    senet_lnH_mse_res = extract_record(senet_lnH_mse_file)

    resnet_H_huber_file = os.path.join("DL_run", "res_file", "check_pt_resnet_100m", "experiment_3", "out_resnet18.txt")
    resnet_H_huber_res = extract_record(resnet_H_huber_file)
    senet_H_huber_file = os.path.join("DL_run", "res_file", "check_pt_senet_100m", "experiment_2", "out_senet18.txt")
    senet_H_huber_res = extract_record(senet_H_huber_file)

    resnet_lnH_huber_file = os.path.join("DL_run", "res_file", "check_pt_resnet_100m", "experiment_5", "out_resnet18.txt")
    resnet_lnH_huber_res = extract_record(resnet_lnH_huber_file)
    senet_lnH_huber_file = os.path.join("DL_run", "res_file", "check_pt_senet_100m", "experiment_4", "out_senet18.txt")
    senet_lnH_huber_res = extract_record(senet_lnH_huber_file)

    # ---plot training and validation loss
    metric = "MAE"

    resnet_lnH_mse_training_loss = resnet_lnH_mse_res["Training"][metric]
    senet_lnH_mse_training_loss = senet_lnH_mse_res["Training"][metric]
    resnet_H_huber_training_loss = resnet_H_huber_res["Training"][metric]
    senet_H_huber_training_loss = senet_H_huber_res["Training"][metric]
    resnet_lnH_huber_training_loss = resnet_lnH_huber_res["Training"][metric]
    senet_lnH_huber_training_loss = senet_lnH_huber_res["Training"][metric]

    resnet_lnH_mse_validation_loss = resnet_lnH_mse_res["Validation"][metric]
    senet_lnH_mse_validation_loss = senet_lnH_mse_res["Validation"][metric]
    resnet_H_huber_validation_loss = resnet_H_huber_res["Validation"][metric]
    senet_H_huber_validation_loss = senet_H_huber_res["Validation"][metric]
    resnet_lnH_huber_validation_loss = resnet_lnH_huber_res["Validation"][metric]
    senet_lnH_huber_validation_loss = senet_lnH_huber_res["Validation"][metric]

    n = 25
    num_epoch = np.arange(1, n+1).astype(np.int)

    fig, ax = plt.subplots(1, 2, figsize=(20, 9))

    ax[0].plot(num_epoch, resnet_lnH_mse_training_loss[0:n], color="royalblue", label="lnH, MSE, ResNet18")
    ax[0].plot(num_epoch, resnet_H_huber_training_loss[0:n], color="mediumblue", label="H, AdaptiveHuber, ResNet18")
    ax[0].plot(num_epoch, resnet_lnH_huber_training_loss[0:n], color="darkblue", label="lnH, AdaptiveHuber, ResNet18")
    ax[0].plot(num_epoch, senet_lnH_mse_training_loss[0:n], color="lightcoral", label="lnH, MSE, SENet18")
    ax[0].plot(num_epoch, senet_H_huber_training_loss[0:n], color="indianred", label="H, AdaptiveHuber, SENet18")
    ax[0].plot(num_epoch, senet_lnH_huber_training_loss[0:n], color="darkred", label="lnH, AdaptiveHuber, SENet18")

    ax[0].set_title("Training {0}".format(metric))
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel(metric)
    ax[0].legend()

    ax[1].plot(num_epoch, resnet_lnH_mse_validation_loss[0:n], color="royalblue", label="lnH, MSE, ResNet18")
    ax[1].plot(num_epoch, resnet_H_huber_validation_loss[0:n], color="mediumblue", label="H, AdaptiveHuber, ResNet18")
    ax[1].plot(num_epoch, resnet_lnH_huber_validation_loss[0:n], color="darkblue", label="lnH, AdaptiveHuber, ResNet18")
    ax[1].plot(num_epoch, senet_lnH_mse_validation_loss[0:n], color="lightcoral", label="lnH, MSE, SENet18")
    ax[1].plot(num_epoch, senet_H_huber_validation_loss[0:n], color="indianred", label="H, AdaptiveHuber, SENet18")
    ax[1].plot(num_epoch, senet_lnH_huber_validation_loss[0:n], color="darkred", label="lnH, AdaptiveHuber, SENet18")

    ax[1].set_title("Validation {0}".format(metric))
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel(metric)
    ax[1].legend()

    # plt.show()
    plt.savefig("train_valid.png", dpi=400)
    '''

    # ---plot confusion matrix
    '''
    #resnet_training_mat = resnet_res["Training"]["confusion_matrix"]
    senet_training_mat = senet_res["Training"]["confusion_matrix"]
    #cbam_training_mat = cbam_res["Training"]["confusion_matrix"]

    #resnet_validation_mat = resnet_res["Validation"]["confusion_matrix"]
    senet_validation_mat = senet_res["Validation"]["confusion_matrix"]
    #cbam_validation_mat = cbam_res["Validation"]["confusion_matrix"]

    num_epoch = np.arange(1, len(senet_training_loss) + 1)

    num_ticks = 10

    for epoch_id in num_epoch:
        fig, ax = plt.subplots(1, figsize=(11, 9))
        c_mat_im = ax.imshow(X=senet_training_mat[:, :, epoch_id-1], cmap="magma_r", vmax=1.0, vmin=0.0)
        ax.set_title("Confusion Matrix of SEResNet at Epoch {0}".format(epoch_id))

        position = fig.add_axes([0.88, 0.11, 0.02, 0.77])
        fig.colorbar(c_mat_im, ax=ax, cax=position)
        # plt.show()
        output_file = os.path.join("DL_run", "res_img", "senet_training_mat_{0}.png".format(epoch_id))
        plt.savefig(output_file, dpi=300)
    '''

    # ---Multi-Task Training
    senet_H_huber_file = os.path.join("DL_run", "res_file", "check_pt_senet_100m_MTL", "experiment_1", "out_senet18_MTL.txt")
    senet_H_huber_res = extract_record_MTL(senet_H_huber_file)

    metric = "MAE"

    senet_H_huber_buildingfootprint_training_loss = senet_H_huber_res["BuildingFootprint"]["Training"][metric]
    senet_H_huber_buildingfootprint_validation_loss = senet_H_huber_res["BuildingFootprint"]["Validation"][metric]
    senet_H_huber_buildingheight_training_loss = senet_H_huber_res["BuildingHeight"]["Training"][metric]
    senet_H_huber_buildingheight_validation_loss = senet_H_huber_res["BuildingHeight"]["Validation"][metric]

    n = 13
    num_epoch = np.arange(1, n + 1).astype(np.int)

    fig, ax = plt.subplots(2, 2, figsize=(16, 14))

    ax[0, 0].plot(num_epoch, senet_H_huber_buildingfootprint_training_loss[0:n], color="royalblue", label="H, AdaptiveHuber, SENet18")
    ax[0, 0].set_title("Training {0} for BuildingFootprint".format(metric))
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].set_ylabel(metric)
    ax[0, 0].legend()

    ax[0, 1].plot(num_epoch, senet_H_huber_buildingfootprint_validation_loss[0:n], color="royalblue", label="H, AdaptiveHuber, SENet18")
    ax[0, 1].set_title("Validation {0} for BuildingFootprint".format(metric))
    ax[0, 1].set_xlabel("Epoch")
    ax[0, 1].set_ylabel(metric)
    ax[0, 1].legend()

    ax[1, 0].plot(num_epoch, senet_H_huber_buildingheight_training_loss[0:n], color="royalblue", label="H, AdaptiveHuber, SENet18")
    ax[1, 0].set_title("Training {0} for BuildingHeight".format(metric))
    ax[1, 0].set_xlabel("Epoch")
    ax[1, 0].set_ylabel(metric)
    ax[1, 0].legend()

    ax[1, 1].plot(num_epoch, senet_H_huber_buildingheight_validation_loss[0:n], color="royalblue", label="H, AdaptiveHuber, SENet18")
    ax[1, 1].set_title("Validation {0} for BuildingHeight".format(metric))
    ax[1, 1].set_xlabel("Epoch")
    ax[1, 1].set_ylabel(metric)
    ax[1, 1].legend()

    # plt.show()
    plt.savefig("train_valid_SENet_MAE.png", dpi=400)