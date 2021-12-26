import argparse
from distutils.core import setup
import shutil
from collections import OrderedDict
from sklearn import metrics
import glob

from .DL_dataset import *
from .DL_model import *


class Evaluator(object):
    def __init__(self, num_quantile, log_scale=False):
        self.sample_pred = []
        self.sample_true = []
        self.num_quantile = num_quantile
        self.log_scale = log_scale

    def r2_score(self):
        true_summary = np.concatenate(self.sample_true, axis=0)
        pred_summary = np.concatenate(self.sample_pred, axis=0)
        r2 = metrics.r2_score(y_true=true_summary, y_pred=pred_summary)
        return r2

    def rmse_score(self):
        true_summary = np.concatenate(self.sample_true, axis=0)
        pred_summary = np.concatenate(self.sample_pred, axis=0)
        rmse = np.sqrt(metrics.mean_squared_error(y_true=true_summary, y_pred=pred_summary))
        return rmse

    def mae_score(self):
        true_summary = np.concatenate(self.sample_true, axis=0)
        pred_summary = np.concatenate(self.sample_pred, axis=0)
        mae = metrics.mean_absolute_error(y_true=true_summary, y_pred=pred_summary)
        return mae

    def record_quantile(self):
        true_summary = np.concatenate(self.sample_true, axis=0)
        pred_summary = np.concatenate(self.sample_pred, axis=0)
        q_list = np.linspace(0.0, 1.0, self.num_quantile + 1)
        q_true = [np.quantile(true_summary, q) for q in q_list]
        q_pred = [np.quantile(pred_summary, q) for q in q_list]
        return q_true, q_pred

    def record_confusion_matrix(self):
        true_summary = np.concatenate(self.sample_true, axis=0)
        pred_summary = np.concatenate(self.sample_pred, axis=0)
        confusion_matrix = get_confusion_matrix(val_true=true_summary, val_pred=pred_summary,
                                                num_quantile=self.num_quantile, normed=True)
        return confusion_matrix

    def add_batch(self, val_true, val_pred):
        self.sample_true.append(val_true)
        self.sample_pred.append(val_pred)

    def update_delta(self, alpha=0.9):
        true_summary = np.concatenate(self.sample_true, axis=0)
        pred_summary = np.concatenate(self.sample_pred, axis=0)

        # ---Note that Evaluator only records H
        # ------when making adjustment for delta, we should be consistent with the model's output
        if self.log_scale:
            true_summary = np.log(true_summary)
            pred_summary = np.log(pred_summary)

        residual = np.abs(true_summary - pred_summary)
        delta = np.quantile(residual, q=alpha)
        return delta

    def reset(self):
        self.sample_true.clear()
        self.sample_pred.clear()


class Evaluator_MTL(object):
    def __init__(self, num_quantile, log_scale=False):
        self.sample_pred = {"footprint": [], "height": []}
        self.sample_true = {"footprint": [], "height": []}
        self.num_quantile = num_quantile
        self.log_scale = log_scale

    def r2_score(self, var_name):
        true_summary = np.concatenate(self.sample_true[var_name], axis=0)
        pred_summary = np.concatenate(self.sample_pred[var_name], axis=0)
        r2 = metrics.r2_score(y_true=true_summary, y_pred=pred_summary)
        return r2

    def rmse_score(self, var_name):
        true_summary = np.concatenate(self.sample_true[var_name], axis=0)
        pred_summary = np.concatenate(self.sample_pred[var_name], axis=0)
        rmse = np.sqrt(metrics.mean_squared_error(y_true=true_summary, y_pred=pred_summary))
        return rmse

    def mae_score(self, var_name):
        true_summary = np.concatenate(self.sample_true[var_name], axis=0)
        pred_summary = np.concatenate(self.sample_pred[var_name], axis=0)
        mae = metrics.mean_absolute_error(y_true=true_summary, y_pred=pred_summary)
        return mae

    def record_quantile(self, var_name):
        true_summary = np.concatenate(self.sample_true[var_name], axis=0)
        pred_summary = np.concatenate(self.sample_pred[var_name], axis=0)
        q_list = np.linspace(0.0, 1.0, self.num_quantile + 1)
        q_true = [np.quantile(true_summary, q) for q in q_list]
        q_pred = [np.quantile(pred_summary, q) for q in q_list]
        return q_true, q_pred

    def record_confusion_matrix(self, var_name):
        true_summary = np.concatenate(self.sample_true[var_name], axis=0)
        pred_summary = np.concatenate(self.sample_pred[var_name], axis=0)
        confusion_matrix = get_confusion_matrix(val_true=true_summary, val_pred=pred_summary,
                                                num_quantile=self.num_quantile, normed=True)
        return confusion_matrix

    def add_batch(self, val_true, val_pred, var_name):
        self.sample_true[var_name].append(val_true)
        self.sample_pred[var_name].append(val_pred)

    def update_delta(self, var_name, alpha=0.9):
        true_summary = np.concatenate(self.sample_true[var_name], axis=0)
        pred_summary = np.concatenate(self.sample_pred[var_name], axis=0)

        if self.log_scale:
            true_summary = np.log(true_summary)
            pred_summary = np.log(pred_summary)

        residual = np.abs(true_summary - pred_summary)
        delta = np.quantile(residual, q=alpha)
        return delta

    def reset(self):
        self.sample_true["footprint"].clear()
        self.sample_true["height"].clear()
        self.sample_pred["footprint"].clear()
        self.sample_pred["height"].clear()


class Trainer(object):
    def __init__(self, args, target_variable, aggregation_list):
        self.args = args
        
        self.aux_namelist = None
        aux_size = None
        if args.aux_feat_info is not None:
            self.aux_namelist = sorted(args.aux_feat_info.keys())
            aux_size = int(args.aux_feat_info[self.aux_namelist[0]] * args.input_size)

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        if target_variable == "BuildingHeight":
            target_id_shift = 2
        elif target_variable == "BuildingFootprint":
            target_id_shift = 1
        else:
            raise NotImplementedError("Unknown target variable")

        # Define Dataloader
        if args.dataset_type == "lmdb":
            self.train_loader = load_data_lmdb(args.training_dataset, args.batch_size, args.num_workers,
                                               aggregation_list, mode="train", cached=args.dataset_cached,
                                               target_id_shift=target_id_shift, log_scale=args.log_scale,
                                               aux_namelist=self.aux_namelist)
            self.val_loader = load_data_lmdb(args.validation_dataset, args.batch_size, args.num_workers,
                                             aggregation_list, mode="valid", cached=args.dataset_cached,
                                             target_id_shift=target_id_shift, log_scale=args.log_scale,
                                             aux_namelist=self.aux_namelist)
        elif args.dataset_type == "hdf5":
            self.train_loader = load_data_hdf5(args.training_dataset, args.batch_size, args.num_workers,
                                               target_variable, aggregation_list, mode="train",
                                               cached=args.dataset_cached, log_scale=args.log_scale,
                                               aux_namelist=self.aux_namelist)
            self.val_loader = load_data_hdf5(args.validation_dataset, args.batch_size, args.num_workers,
                                             target_variable, aggregation_list, mode="valid",
                                             cached=args.dataset_cached, log_scale=args.log_scale,
                                             aux_namelist=self.aux_namelist)
        else:
            raise NotImplementedError

        # Define prediction model
        if args.input_size == 15:
            in_plane = 64
            num_block = 2
        elif args.input_size == 30:
            in_plane = 64
            num_block = 1
        elif args.input_size == 60:
            in_plane = 64
            num_block = 1
        else:
            in_plane = 64
            num_block = 1

        if target_variable == "BuildingHeight":
            activation = "relu"
        elif target_variable == "BuildingFootprint":
            activation = "sigmoid"
        else:
            raise NotImplementedError("Unknown target variable")

        if args.model == "ResNet18":
            if self.aux_namelist is None:
                self.model = model_ResNet(in_plane=in_plane, input_channels=6, input_size=args.input_size,
                                            num_block=num_block, log_scale=args.log_scale, activation=activation,
                                            cuda_used=self.args.cuda, trained_record=args.trained_record)
            else:
                self.model = model_ResNet_aux(in_plane=in_plane, input_channels=6, input_size=args.input_size,
                                                aux_input_size=aux_size, num_aux=len(self.aux_namelist),
                                                num_block=num_block, log_scale=args.log_scale, activation=activation,
                                                cuda_used=self.args.cuda, trained_record=args.trained_record)
        elif args.model == "SEResNet18":
            if self.aux_namelist is None:
                self.model = model_SEResNet(in_plane=in_plane, input_channels=6, input_size=args.input_size,
                                            num_block=num_block, log_scale=args.log_scale, activation=activation,
                                            cuda_used=self.args.cuda, trained_record=args.trained_record)
            else:
                self.model = model_SEResNet_aux(in_plane=in_plane, input_channels=6, input_size=args.input_size,
                                                aux_input_size=aux_size, num_aux=len(self.aux_namelist),
                                                num_block=num_block, log_scale=args.log_scale, activation=activation,
                                                cuda_used=self.args.cuda, trained_record=args.trained_record)
        elif args.model == "CBAMResNet18":
            if self.aux_namelist is None:
                self.model = model_CBAMResNet(in_plane=in_plane, input_channels=6, input_size=args.input_size,
                                                num_block=num_block, log_scale=args.log_scale, activation=activation,
                                                cuda_used=self.args.cuda, trained_record=args.trained_record)
            else:
                self.model = model_CBAMResNet_aux(in_plane=in_plane, input_channels=6, input_size=args.input_size,
                                                    aux_input_size=aux_size, num_aux=len(self.aux_namelist),
                                                    num_block=num_block, log_scale=args.log_scale, activation=activation,
                                                    cuda_used=self.args.cuda, trained_record=args.trained_record)
        else:
            raise NotImplementedError

        # Define Optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # Define Criterion
        # whether to use class balanced weights
        self.criterion = RegressionLosses(cuda=args.cuda).build_loss(loss_mode=args.loss_type)

        # Define Evaluator
        self.evaluator = Evaluator(num_quantile=args.num_quantile, log_scale=args.log_scale)
        # Define lr scheduler
        if args.lr_scheduler == "step":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif args.lr_scheduler == "poly":
            poly_lr = lambda epoch: (1.0-epoch/args.epochs)**0.9
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=poly_lr)
        elif args.lr_scheduler == "cos":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2, eta_min=args.lr/1000.0)

        # Using cuda
        if self.args.cuda:
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0

    def train(self, epoch):
        self.model.train()
        self.evaluator.reset()
        train_loss = 0.0
        for i, sample in enumerate(self.train_loader):
            if self.aux_namelist is None:
                input_band, target = sample["feature"], sample["value"]
                if self.args.cuda:
                    input_band, target = input_band.cuda(), target.cuda()
                self.optimizer.zero_grad()
                output = self.model(input_band)
            else:
                input_band, target, aux_feat = sample["feature"], sample["value"], sample["aux_feature"]
                if self.args.cuda:
                    input_band, target, aux_feat = input_band.cuda(), target.cuda(), aux_feat.cuda()
                self.optimizer.zero_grad()
                output = self.model(input_band, aux_feat)

            output = torch.squeeze(output)
            loss = self.criterion(output, target, beta=self.args.delta)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() * input_band.size(0)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()

            if self.args.log_scale:
                # Add batch sample into evaluator with conversion from ln(H) to H
                self.evaluator.add_batch(np.exp(target), np.exp(pred))
            else:
                self.evaluator.add_batch(target, pred)

        r2 = self.evaluator.r2_score()
        rmse = self.evaluator.rmse_score()
        mae = self.evaluator.mae_score()
        q_true, q_pred = self.evaluator.record_quantile()
        confusion_matrix = self.evaluator.record_confusion_matrix()
        print('-----Training-----')
        print('Training Loss at Epoch ', epoch, ': %.4f' % train_loss)
        print('Training R^2 at Epoch ', epoch, ': %.4f' % r2)
        print('Training RMSE at Epoch ', epoch, ': %.4f' % rmse)
        print('Training MAE at Epoch ', epoch, ': %.4f' % mae)
        print('Training Quantiles of Target Distribution at Epoch ', epoch, ': ')
        print(" ".join(map(lambda x: "%.3f" % x, q_true)))
        print('Training Quantiles of Prediction Distribution at Epoch ', epoch, ': ')
        print(" ".join(map(lambda x: "%.3f" % x, q_pred)))
        print('Training confusion matrix at Epoch ', epoch, ': ')
        for i in range(0, self.args.num_quantile + 2):
            print(" ".join(map(lambda x: "%.3f" % x, confusion_matrix[i])))

        if self.args.loss_type == "AdaptiveHuber":
            self.args.delta = self.evaluator.update_delta(alpha=self.args.alpha)
            print('Training delta value for AdaptiveHuberLoss at Epoch ', epoch, ': %.4f' % self.args.delta)

    def valid(self, epoch, end_flag=False):
        self.model.eval()
        self.evaluator.reset()
        valid_loss = 0.0
        for i, sample in enumerate(self.val_loader):
            if self.aux_namelist is None:
                input_band, target = sample["feature"], sample["value"]
                if self.args.cuda:
                    input_band, target = input_band.cuda(), target.cuda()
                with torch.no_grad():
                    output = self.model(input_band)
            else:
                input_band, target, aux_feat = sample["feature"], sample["value"], sample["aux_feature"]
                if self.args.cuda:
                    input_band, target, aux_feat = input_band.cuda(), target.cuda(), aux_feat.cuda()
                with torch.no_grad():
                    output = self.model(input_band, aux_feat)
            
            output = torch.squeeze(output)
            loss = self.criterion(output, target, beta=self.args.delta)

            valid_loss += loss.item() * input_band.size(0)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()

            if self.args.log_scale:
                # Add batch sample into evaluator with conversion from ln(H) to H
                self.evaluator.add_batch(np.exp(target), np.exp(pred))
            else:
                self.evaluator.add_batch(target, pred)

        r2 = self.evaluator.r2_score()
        rmse = self.evaluator.rmse_score()
        mae = self.evaluator.mae_score()
        q_true, q_pred = self.evaluator.record_quantile()
        confusion_matrix = self.evaluator.record_confusion_matrix()
        print('-----Validation-----')
        print('Validation Loss at Epoch ', epoch, ': %.4f' % valid_loss)
        print('Validation R^2 at Epoch ', epoch, ': %.4f' % r2)
        print('Validation RMSE at Epoch ', epoch, ': %.4f' % rmse)
        print('Validation MAE at Epoch ', epoch, ': %.4f' % mae)
        print('Validation Quantiles of Target Distribution at Epoch ', epoch, ': ')
        print(" ".join(map(lambda x: "%.3f" % x, q_true)))
        print('Validation Quantiles of Prediction Distribution at Epoch ', epoch, ': ')
        print(" ".join(map(lambda x: "%.3f" % x, q_pred)))
        print('Validation confusion matrix at Epoch ', epoch, ': ')
        for i in range(0, self.args.num_quantile+2):
            print(" ".join(map(lambda x: "%.3f" % x, confusion_matrix[i])))

        new_pred = r2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)
        if end_flag:
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'end_pred': new_pred,
            }, is_best, filename='end_checkpoint.pth.tar')


class TrainerMTL(object):
    def __init__(self, args, aggregation_list):
        self.args = args

        self.aux_namelist = None
        aux_size = None
        if args.aux_feat_info is not None:
            self.aux_namelist = sorted(args.aux_feat_info.keys())
            aux_size = int(args.aux_feat_info[self.aux_namelist[0]] * args.input_size)

        # Define Saver
        self.saver = Saver_MTL(args)
        self.saver.save_experiment_config()

        # Define Dataloader
        if args.dataset_type == "lmdb":
            self.train_loader = load_data_lmdb_MTL(args.training_dataset, args.batch_size, args.num_workers,
                                                   aggregation_list, mode="train", cached=args.dataset_cached,
                                                   log_scale=args.log_scale, aux_namelist=self.aux_namelist)
            self.val_loader = load_data_lmdb_MTL(args.validation_dataset, args.batch_size, args.num_workers,
                                                 aggregation_list, mode="valid", cached=args.dataset_cached,
                                                 log_scale=args.log_scale, aux_namelist=self.aux_namelist)
        else:
            raise NotImplementedError

        if args.input_size == 15:
            in_plane = 64
            num_block = 2
        elif args.input_size == 30:
            in_plane = 64
            num_block = 1
        elif args.input_size == 60:
            in_plane = 64
            num_block = 1
        else:
            in_plane = 64
            num_block = 1

        # Define prediction model
        if args.model == "ResNet18":
            if self.aux_namelist is None:
                self.model = model_ResNetMTL(in_plane=in_plane, input_channels=6, input_size=args.input_size,
                                                num_block=num_block, log_scale=args.log_scale, crossed=args.MTL_crossed,
                                                cuda_used=self.args.cuda, trained_record=args.trained_record)
            else:
                self.model = model_ResNetMTL_aux(in_plane=in_plane, input_channels=6, input_size=args.input_size,
                                                    aux_input_size=aux_size, num_aux=len(self.aux_namelist),
                                                    num_block=num_block, log_scale=args.log_scale, crossed=args.MTL_crossed,
                                                    cuda_used=self.args.cuda, trained_record=args.trained_record)
        elif args.model == "SEResNet18":
            if self.aux_namelist is None:
                self.model = model_SEResNetMTL(in_plane=in_plane, input_channels=6, input_size=args.input_size,
                                                num_block=num_block, log_scale=args.log_scale, crossed=args.MTL_crossed,
                                                cuda_used=self.args.cuda, trained_record=args.trained_record)
            else:
                self.model = model_SEResNetMTL_aux(in_plane=in_plane, input_channels=6, input_size=args.input_size,
                                                    aux_input_size=aux_size, num_aux=len(self.aux_namelist),
                                                    num_block=num_block, log_scale=args.log_scale, crossed=args.MTL_crossed,
                                                    cuda_used=self.args.cuda, trained_record=args.trained_record)
        elif args.model == "CBAMResNet18":
            if self.aux_namelist is None:
                self.model = model_CBAMResNetMTL(in_plane=in_plane, input_channels=6, input_size=args.input_size,
                                                    num_block=num_block, log_scale=args.log_scale, crossed=args.MTL_crossed,
                                                    cuda_used=self.args.cuda, trained_record=args.trained_record)
            else:
                self.model = model_CBAMResNetMTL_aux(in_plane=in_plane, input_channels=6, input_size=args.input_size,
                                                        aux_input_size=aux_size, num_aux=len(self.aux_namelist),
                                                        num_block=num_block, log_scale=args.log_scale, crossed=args.MTL_crossed,
                                                        cuda_used=self.args.cuda, trained_record=args.trained_record)
        else:
            raise NotImplementedError

        # Define Criterion
        if args.weight_1 is None or args.weight_2 is None:
            # ---whether to use adaptive weights
            adaptive_weight = True
            if self.args.cuda:
                self.s_footprint = torch.zeros((1,), requires_grad=True, device="cuda")
                self.s_height = torch.zeros((1,), requires_grad=True, device="cuda")
            else:
                self.s_footprint = torch.zeros((1,), requires_grad=True)
                self.s_height = torch.zeros((1,), requires_grad=True)
            params = ([p for p in self.model.parameters()] + [self.s_footprint, self.s_height])
        else:
            adaptive_weight = False
            self.s_footprint = args.weight_1
            self.s_height = args.weight_2
            params = ([p for p in self.model.parameters()])

        self.criterion = RegressionLosses_MTL(cuda=args.cuda, adaptive_weight=adaptive_weight).build_loss(loss_mode=args.loss_type)

        # Define Optimizer
        self.optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

        # Define Evaluator
        self.evaluator = Evaluator_MTL(num_quantile=args.num_quantile, log_scale=args.log_scale)

        # Define lr scheduler
        if args.lr_scheduler == "step":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif args.lr_scheduler == "poly":
            poly_lr = lambda epoch: (1.0-epoch/args.epochs)**0.9
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=poly_lr)
        elif args.lr_scheduler == "cos":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2, eta_min=args.lr/1000.0)

        # Using cuda
        if self.args.cuda:
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred_height = -1.0
        self.best_pred_footprint = -1.0

    def train(self, epoch):
        self.model.train()
        self.evaluator.reset()
        train_loss = 0.0
        for i, sample in enumerate(self.train_loader):
            if self.aux_namelist is None:
                input_band, target_footprint, target_height = sample["feature"], sample["footprint"], sample["height"]
                if self.args.cuda:
                    input_band, target_footprint, target_height = input_band.cuda(), target_footprint.cuda(), target_height.cuda()
                self.optimizer.zero_grad()
                output_footprint, output_height = self.model(input_band)
            else:
                input_band, aux_feat, target_footprint, target_height = sample["feature"], sample["aux_feature"], sample["footprint"], sample["height"]
                if self.args.cuda:
                    input_band, aux_feat, target_footprint, target_height = input_band.cuda(), aux_feat.cuda(), target_footprint.cuda(), target_height.cuda()
                self.optimizer.zero_grad()
                output_footprint, output_height = self.model(input_band, aux_feat)

            output_footprint = torch.squeeze(output_footprint)
            output_height = torch.squeeze(output_height)
            loss = self.criterion(footprint_pred=output_footprint, footprint_true=target_footprint,
                                  height_pred=output_height, height_true=target_height,
                                  beta_footprint=self.args.delta, beta_height=self.args.delta_back,
                                  s_footprint=self.s_footprint, s_height=self.s_height)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() * input_band.size(0)
            pred_footprint = output_footprint.data.cpu().numpy()
            target_footprint = target_footprint.cpu().numpy()
            pred_height = output_height.data.cpu().numpy()
            target_height = target_height.cpu().numpy()

            if self.args.log_scale:
                # Add batch sample into evaluator with conversion from ln(H) to H
                self.evaluator.add_batch(np.exp(target_height), np.exp(pred_height), var_name="height")
            else:
                self.evaluator.add_batch(target_height, pred_height, var_name="height")

            self.evaluator.add_batch(target_footprint, pred_footprint, var_name="footprint")

        r2_height = self.evaluator.r2_score(var_name="height")
        rmse_height = self.evaluator.rmse_score(var_name="height")
        mae_height = self.evaluator.mae_score(var_name="height")
        q_true_height, q_pred_height = self.evaluator.record_quantile(var_name="height")
        confusion_matrix_height = self.evaluator.record_confusion_matrix(var_name="height")
        r2_footprint = self.evaluator.r2_score(var_name="footprint")
        rmse_footprint = self.evaluator.rmse_score(var_name="footprint")
        mae_footprint = self.evaluator.mae_score(var_name="footprint")
        q_true_footprint, q_pred_footprint = self.evaluator.record_quantile(var_name="footprint")
        confusion_matrix_footprint = self.evaluator.record_confusion_matrix(var_name="footprint")
        print('-----Training-----')
        print('Training Loss at Epoch ', epoch, ': %.4f' % train_loss)
        # ------report the performance of building height prediction
        print('Training R^2 of BuildingHeight at Epoch ', epoch, ': %.4f' % r2_height)
        print('Training RMSE of BuildingHeight at Epoch ', epoch, ': %.4f' % rmse_height)
        print('Training MAE of BuildingHeight at Epoch ', epoch, ': %.4f' % mae_height)
        print('Training Quantiles of Target Distribution of BuildingHeight at Epoch ', epoch, ': ')
        print(" ".join(map(lambda x: "%.3f" % x, q_true_height)))
        print('Training Quantiles of Prediction Distribution of BuildingHeight at Epoch ', epoch, ': ')
        print(" ".join(map(lambda x: "%.3f" % x, q_pred_height)))
        print('Training confusion matrix of BuildingHeight at Epoch ', epoch, ': ')
        for i in range(0, args.num_quantile + 2):
            print(" ".join(map(lambda x: "%.3f" % x, confusion_matrix_height[i])))
        # ------report the performance of building footprint prediction
        print('Training R^2 of BuildingFootprint at Epoch ', epoch, ': %.4f' % r2_footprint)
        print('Training RMSE of BuildingFootprint at Epoch ', epoch, ': %.4f' % rmse_footprint)
        print('Training MAE of BuildingFootprint at Epoch ', epoch, ': %.4f' % mae_footprint)
        print('Training Quantiles of Target Distribution of BuildingFootprint at Epoch ', epoch, ': ')
        print(" ".join(map(lambda x: "%.3f" % x, q_true_footprint)))
        print('Training Quantiles of Prediction Distribution of BuildingFootprint at Epoch ', epoch, ': ')
        print(" ".join(map(lambda x: "%.3f" % x, q_pred_footprint)))
        print('Training confusion matrix of BuildingFootprint at Epoch ', epoch, ': ')
        for i in range(0, args.num_quantile + 2):
            print(" ".join(map(lambda x: "%.3f" % x, confusion_matrix_footprint[i])))
        # ------report the weighting parameters
        if type(self.s_footprint) is torch.Tensor:
            w_footprint = 0.5 * torch.exp(-self.s_footprint)
            w_height = 0.5 * torch.exp(-self.s_height)
            print('Weighting Parameter for BuildingFootprint Loss at Epoch ', epoch, ': %.4f' % w_footprint.item())
            print('Weighting Parameter for BuildingHeight Loss at Epoch ', epoch, ': %.4f' % w_height.item())

        if self.args.loss_type == "AdaptiveHuber":
            self.args.delta = self.evaluator.update_delta(alpha=self.args.alpha, var_name="footprint")
            print('Training delta value of BuildingFootprint for AdaptiveHuberLoss at Epoch ', epoch, ': %.4f' % self.args.delta)
            self.args.delta_back = self.evaluator.update_delta(alpha=self.args.alpha, var_name="height")
            print('Training delta value of BuildingHeight for AdaptiveHuberLoss at Epoch ', epoch, ': %.4f' % self.args.delta_back)

    def valid(self, epoch, end_flag=False):
        self.model.eval()
        self.evaluator.reset()
        valid_loss = 0.0
        for i, sample in enumerate(self.val_loader):
            if self.aux_namelist is None:
                input_band, target_footprint, target_height = sample["feature"], sample["footprint"], sample["height"]
                if self.args.cuda:
                    input_band, target_footprint, target_height = input_band.cuda(), target_footprint.cuda(), target_height.cuda()
                with torch.no_grad():
                    output_footprint, output_height = self.model(input_band)
            else:
                input_band, aux_feat, target_footprint, target_height = sample["feature"], sample["aux_feature"], sample["footprint"], sample["height"]
                if self.args.cuda:
                    input_band, aux_feat, target_footprint, target_height = input_band.cuda(), aux_feat.cuda(), target_footprint.cuda(), target_height.cuda()
                with torch.no_grad():
                    output_footprint, output_height = self.model(input_band, aux_feat)
                    
            output_footprint = torch.squeeze(output_footprint)
            output_height = torch.squeeze(output_height)
            loss = self.criterion(footprint_pred=output_footprint, footprint_true=target_footprint,
                                  height_pred=output_height, height_true=target_height,
                                  beta_footprint=self.args.delta, beta_height=self.args.delta_back,
                                  s_footprint=self.s_footprint, s_height=self.s_height)

            valid_loss += loss.item() * input_band.size(0)
            pred_footprint = output_footprint.data.cpu().numpy()
            target_footprint = target_footprint.cpu().numpy()
            pred_height = output_height.data.cpu().numpy()
            target_height = target_height.cpu().numpy()

            if self.args.log_scale:
                # Add batch sample into evaluator with conversion from ln(H) to H
                self.evaluator.add_batch(np.exp(target_height), np.exp(pred_height), var_name="height")
            else:
                self.evaluator.add_batch(target_height, pred_height, var_name="height")

            self.evaluator.add_batch(target_footprint, pred_footprint, var_name="footprint")

        r2_height = self.evaluator.r2_score(var_name="height")
        rmse_height = self.evaluator.rmse_score(var_name="height")
        mae_height = self.evaluator.mae_score(var_name="height")
        q_true_height, q_pred_height = self.evaluator.record_quantile(var_name="height")
        confusion_matrix_height = self.evaluator.record_confusion_matrix(var_name="height")
        r2_footprint = self.evaluator.r2_score(var_name="footprint")
        rmse_footprint = self.evaluator.rmse_score(var_name="footprint")
        mae_footprint = self.evaluator.mae_score(var_name="footprint")
        q_true_footprint, q_pred_footprint = self.evaluator.record_quantile(var_name="footprint")
        confusion_matrix_footprint = self.evaluator.record_confusion_matrix(var_name="footprint")
        print('-----Validation-----')
        print('Validation Loss at Epoch ', epoch, ': %.4f' % valid_loss)
        # ------report the performance of building height prediction
        print('Validation R^2 of BuildingHeight at Epoch ', epoch, ': %.4f' % r2_height)
        print('Validation RMSE of BuildingHeight at Epoch ', epoch, ': %.4f' % rmse_height)
        print('Validation MAE of BuildingHeight at Epoch ', epoch, ': %.4f' % mae_height)
        print('Validation Quantiles of Target Distribution of BuildingHeight at Epoch ', epoch, ': ')
        print(" ".join(map(lambda x: "%.3f" % x, q_true_height)))
        print('Validation Quantiles of Prediction Distribution of BuildingHeight at Epoch ', epoch, ': ')
        print(" ".join(map(lambda x: "%.3f" % x, q_pred_height)))
        print('Validation confusion matrix of BuildingHeight at Epoch ', epoch, ': ')
        for i in range(0, self.args.num_quantile + 2):
            print(" ".join(map(lambda x: "%.3f" % x, confusion_matrix_height[i])))
        # ------report the performance of building footprint prediction
        print('Validation R^2 of BuildingFootprint at Epoch ', epoch, ': %.4f' % r2_footprint)
        print('Validation RMSE of BuildingFootprint at Epoch ', epoch, ': %.4f' % rmse_footprint)
        print('Validation MAE of BuildingFootprint at Epoch ', epoch, ': %.4f' % mae_footprint)
        print('Validation Quantiles of Target Distribution of BuildingFootprint at Epoch ', epoch, ': ')
        print(" ".join(map(lambda x: "%.3f" % x, q_true_footprint)))
        print('Validation Quantiles of Prediction Distribution of BuildingFootprint at Epoch ', epoch, ': ')
        print(" ".join(map(lambda x: "%.3f" % x, q_pred_footprint)))
        print('Validation confusion matrix of BuildingFootprint at Epoch ', epoch, ': ')
        for i in range(0, self.args.num_quantile + 2):
            print(" ".join(map(lambda x: "%.3f" % x, confusion_matrix_footprint[i])))

        new_pred_height = r2_height
        new_pred_footprint = r2_footprint
        if (new_pred_height > self.best_pred_height) and (new_pred_footprint > self.best_pred_footprint):
            is_best = True
            self.best_pred_height = new_pred_height
            self.best_pred_footprint = new_pred_footprint
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_pred_footprint': self.best_pred_footprint,
                'best_pred_height': self.best_pred_height,
            }, is_best)
        if end_flag:
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'end_pred_footprint': new_pred_footprint,
                'end_pred_height': new_pred_height,
            }, is_best, filename='end_checkpoint.pth.tar')


class Saver(object):
    def __init__(self, args):
        self.args = args
        self.directory = os.path.join(args.base_dir, args.checkname)
        # check the experiments have been run so far
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')), key=lambda x: int(x.split('_')[-1]))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred) + "\n")
            if self.runs:
                previous_r2 = [-100.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            r2 = float(f.readline())
                            previous_r2.append(r2)
                    else:
                        continue
                max_r2 = max(previous_r2)
                if best_pred > max_r2:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
        else:
            end_pred = state['end_pred']
            with open(os.path.join(self.experiment_dir, 'end_pred.txt'), 'w') as f:
                f.write(str(end_pred) + "\n")

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['model'] = self.args.model
        p['train_set'] = self.args.training_dataset
        p['valid_set'] = self.args.validation_dataset
        p['log_scale'] = self.args.log_scale
        p['loss_type'] = self.args.loss_type
        p['delta'] = self.args.delta
        p['alpha'] = self.args.alpha
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['batch_size'] = self.args.batch_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()


class Saver_MTL(object):
    def __init__(self, args):
        self.args = args
        self.directory = os.path.join(args.base_dir, args.checkname)
        # check the experiments have been run so far
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')), key=lambda x: int(x.split('_')[-1]))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred_footprint = state['best_pred_footprint']
            best_pred_height = state['best_pred_height']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred_footprint) + "\n")
                f.write(str(best_pred_height) + "\n")
            if self.runs:
                previous_r2_footprint = [-100.0]
                previous_r2_height = [-100.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            lines = f.readlines()
                            r2_footprint = float(lines[0])
                            r2_height = float(lines[1])
                            previous_r2_footprint.append(r2_footprint)
                            previous_r2_height.append(r2_height)
                    else:
                        continue
                max_r2_footprint = max(previous_r2_footprint)
                max_r2_height = max(previous_r2_height)
                if (best_pred_footprint > max_r2_footprint) and (best_pred_height > max_r2_height):
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
        else:
            end_pred_footprint = state['end_pred_footprint']
            end_pred_height = state['end_pred_height']
            with open(os.path.join(self.experiment_dir, 'end_pred.txt'), 'w') as f:
                f.write(str(end_pred_footprint) + "\n")
                f.write(str(end_pred_height) + "\n")

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['model'] = self.args.model
        p['train_set'] = self.args.training_dataset
        p['valid_set'] = self.args.validation_dataset
        p['log_scale'] = self.args.log_scale
        p['loss_type'] = self.args.loss_type
        p['delta'] = self.args.delta
        p['delta_back'] = self.args.delta_back
        p['alpha'] = self.args.alpha
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['batch_size'] = self.args.batch_size
        p['MTL_crossed'] = self.args.MTL_crossed
        if self.args.weight_1 is None or self.args.weight_2 is None:
            p['MTL_dynamic_weight'] = True
        else:
            p['MTL_dynamic_weight'] = False

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Model Training")
    # dataset
    parser.add_argument("--target_variable", type=str, default="BuildingHeight",
                        choices=["BuildingHeight", "BuildingFootprint"], help="variable to be predicted (ignored by MTL)")
    parser.add_argument("--training_dataset", type=str, default="dataset/patch_data_50pt_s15_sample_train.lmdb",
                        help="path of the training dataset")
    parser.add_argument("--validation_dataset", type=str, default="dataset/patch_data_50pt_s15_sample_valid.lmdb",
                        help="path of the validation dataset")
    parser.add_argument("--dataset_type", type=str, default="lmdb", choices=["lmdb", "hdf5"],
                        help="type of input dataset (default: lmdb)")
    parser.add_argument("--dataset_cached", type=str, default="True",
                        help="determine whether the whole dataset would be cached into memory")
    parser.add_argument("--input_size", type=int, default=15, help="input image size")
    parser.add_argument("--log_scale", type=str, default="False", choices=["True", "False"],
                        help="determine whether the prediction result of DNNs is ln(H)")
    parser.add_argument("--aux_feature", type=str, default=None, help="comma-separated namelist of auxiliary features (e.g. DEM) for prediction")
    parser.add_argument("--aux_patch_size_ratio", type=str, default=None, 
                        help="comma-separated list of the patch size ratio between auxiliary data and Sentinel's data")

    # training hyper params
    # ---[1] CNN backbone
    parser.add_argument("--model", type=str, default="ResNet18", choices=["ResNet18", "SEResNet18", "CBAMResNet18"],
                        help="Deep Neural Networks model for building information extraction (default: ResNet18)")
    parser.add_argument("--trained_record", type=str, default=None, help="pretrained weights for CNN initialization")
    # ---[2] loss function
    parser.add_argument("--loss_type", type=str, default="MSE", choices=["MSE", "Huber", "AdaptiveHuber"],
                        help="loss func (default: MSE)")
    parser.add_argument("--delta", type=float, default=1.0,
                        help="delta value used for Huber Loss which can be estimated from the Median Absolute Deviation")
    parser.add_argument("--delta_back", type=float, default=1.0,
                        help="delta value used in Multi-Task Learning for another variable if HuberLoss is chosen")
    parser.add_argument("--alpha", type=float, default=0.9,
                        help="alpha-quantile of model residual used for adaptive delta estimation for the Huber Loss")
    # ---[3] optimizer
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("--lr_scheduler", type=str, default="cos", choices=["poly", "step", "cos"],
                        help="lr scheduler mode: (default: cos)")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum (default: 0.9)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="w-decay (default: 1e-4)")

    # ---[4] Multi-Task Learning
    parser.add_argument("--MTL", type=str, default="False", choices=["True", "False"],
                        help="determine whether Multi-Task Learning is used")
    parser.add_argument("--weight_1", type=float, default=None, help="building footprint loss weighting")
    parser.add_argument("--weight_2", type=float, default=None, help="building height loss weighting")
    parser.add_argument("--MTL_crossed", type=str, default="False", choices=["True", "False"],
                        help="determine whether the hidden states of height and footprint are multiplied")
    # ---[5] evaluation metric
    parser.add_argument("--num_quantile", type=int, default=20,
                        help="number of quantiles used for mapping continuous target into its categorical form")
    # ---[6] other
    parser.add_argument("--epochs", type=int, default=130, help="number of epochs to train (default: 130)")
    parser.add_argument("--early_stopping", type=int, default=None, help="stop the model training early")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for training (default: 8)")
    parser.add_argument("--num_workers", type=int, default=1, help="number of workers used for data load (default: 2)")

    # cuda, seed and logging
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--gpu-ids", type=str, default='0',
                        help="use which gpu to train, must be a comma-separated list of integers only (default=0)")
    parser.add_argument("--seed", type=int, default=117, help="random seed (default: 116)")
    parser.add_argument("--base_dir", type=str, default="DL_run", help="base directory for experiment running")
    parser.add_argument("--checkname", type=str, default="check_pt", help="set the checkpoint name")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(",")]
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids[0])
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
    
    args.aux_feat_info = None
    if args.aux_feature is not None:
        args.aux_feature = [s for s in args.aux_feature.split(",")]
        args.aux_patch_size_ratio = [int(s) for s in args.aux_patch_size_ratio.split(",")]
        if not all(args.aux_patch_size_ratio[0] == s for s in args.aux_patch_size_ratio):
            raise Exception("All of auxiliary data patch should be of the same size.")
        args.aux_feat_info = {args.aux_feature[i]: args.aux_patch_size_ratio[i] for i in range(0, len(args.aux_feature))}

    if args.MTL == "True":
        args.MTL = True
    else:
        args.MTL = False

    if args.MTL_crossed == "True":
        args.MTL_crossed = True
    else:
        args.MTL_crossed = False

    if args.log_scale == "True":
        args.log_scale = True
    else:
        args.log_scale = False

    if args.dataset_cached == "True":
        args.dataset_cached = True
    else:
        args.dataset_cached = False

    print(args)
    torch.manual_seed(args.seed)

    # torch.set_num_threads(16)

    if args.MTL:
        trainer = TrainerMTL(args, aggregation_list=["50pt"])
    else:
        trainer = Trainer(args, target_variable=args.target_variable, aggregation_list=["50pt"])

    print('Total Epoches:', trainer.args.epochs)

    for epoch in range(0, trainer.args.epochs):
        '''
        # early stopping and restart at the previous best point
        if args.early_stopping is not None:
            if epoch == args.early_stopping:
                break
        '''

        print('*' * 50)

        trainer.train(epoch)
        if epoch == trainer.args.epochs - 1:
            trainer.valid(epoch, end_flag=True)
        else:
            trainer.valid(epoch, end_flag=False)

        trainer.lr_scheduler.step()
