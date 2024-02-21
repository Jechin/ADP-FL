'''
Description: 
Author: Jechin jechinyu@163.com
Date: 2024-02-16 16:15:59
LastEditors: Jechin jechinyu@163.com
LastEditTime: 2024-02-21 14:15:33
'''
import torch
from torch import nn, autograd
# from utils.dp_mechanism import cal_sensitivity, cal_sensitivity_MA, Laplace, Gaussian_Simple, Gaussian_MA
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import copy
from utils.loss import DiceLoss
from utils.util import _eval_haus, _eval_iou

def metric_calc(gt, pred, score):
    tn, fp, fn, tp = metrics.confusion_matrix(gt, pred).ravel()
    acc = metrics.accuracy_score(gt, pred)
    try:
        auc = metrics.roc_auc_score(gt, score)
    except ValueError:
        auc = 0
    sen = metrics.recall_score(gt, pred)  # recall = sensitivity = TP/TP+FN
    spe = tn / (tn + fp)  # specificity = TN / (TN+FP)
    f1 = metrics.f1_score(gt, pred)
    return [tn, fp, fn, tp], auc, acc, sen, spe, f1

class LocalUpdateDP(object):
    def __init__(self, args, train_loader, val_loader, test_loader, loss_fun, device, logging, idx) -> None:
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fun = loss_fun
        self.lr = args.lr
        self.device = device
        self.logging = logging
        self.idx = idx

    def train(self, model, sigma=None):
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)
        loss_all = 0
        segmentation = "UNet" in model.__class__.__name__
        train_acc = 0.0 if not segmentation else {}
        model_pred, label_gt, pred_prob = [], [], []
        num_sample_test = 0
        origin_model = copy.deepcopy(model).to("cpu")
        w_k_2 = None
        w_k_1 = None

        for step, data in enumerate(self.train_loader):
            if step == len(self.train_loader) - 2:
                w_k_2 = copy.deepcopy(model).to("cpu")
            if step == len(self.train_loader) - 1:
                w_k_1 = copy.deepcopy(model).to("cpu")
                
            if self.args.data.startswith("prostate"):
                inp = data["Image"]
                target = data["Mask"]
                target = target.to(self.device)
            else:
                inp = data["Image"]
                target = data["Label"]
                target = target.to(self.device)
            
            model.to(self.device)
            model.zero_grad()
            inp = inp.to(self.device)
            output = model(inp)
            
            if self.args.data.startswith("prostate"):
                loss = self.loss_fun(output[:, 0, :, :], target)
            else:
                loss = self.loss_fun(output, target)

            if segmentation:
                if self.args.data.startswith("prostate"):
                    if len(train_acc.keys()) == 0:
                        train_acc["Dice"] = DiceLoss().dice_coef(output[:, 0, :, :], target).item()
                        train_acc["IoU"] = _eval_iou(output[:, 0, :, :], target).item()
                        train_acc["HD"] = 0.0
                    else:
                        train_acc["Dice"] += DiceLoss().dice_coef(output[:, 0, :, :], target).item()
                        train_acc["IoU"] += _eval_iou(output[:, 0, :, :], target).item()

                    for i_b in range(output.shape[0]):
                        hd = _eval_haus(output[i_b, 0, :, :], target[i_b]).item()
                        if hd > 0:
                            train_acc["HD"] += hd
                            num_sample_test += 1
                else:
                    if len(train_acc.keys()) == 0:
                        train_acc["Dice"] = DiceLoss().dice_coef(output, target).item()
                    else:
                        train_acc["Dice"] += DiceLoss().dice_coef(output, target).item()
            else:
                out_prob = torch.nn.functional.softmax(output, dim=1)
                model_pred.extend(out_prob.data.max(1)[1].view(-1).detach().cpu().numpy())
                pred_prob.extend(out_prob.data[:, 1].view(-1).detach().cpu().numpy())
                label_gt.extend(target.view(-1).detach().cpu().numpy())

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            loss_all += loss.item()

        loss = loss_all / len(self.train_loader)
        if segmentation:
            acc = {
                "Dice": train_acc["Dice"] / len(self.train_loader),
                "IoU": train_acc["IoU"] / len(self.train_loader),
                "HD": train_acc["HD"] / num_sample_test,
            }
        else:
            model_pred = np.asarray(model_pred)
            pred_prob = np.asarray(pred_prob)
            label_gt = np.asarray(label_gt)
            metric_res = metric_calc(label_gt, model_pred, pred_prob)
            acc = {
                "AUC": metric_res[1],
                "Acc": metric_res[2],
                "Sen": metric_res[3],
                "Spe": metric_res[4],
                "F1": metric_res[5],
            }
        
        self.lr = scheduler.get_last_lr()[0]

        out_str = ""
        for k, v in acc.items():
            out_str += " | Train {}: {:.4f} ".format(k, v)
        self.logging.info(
            "Site-{:<5s} rounds:{:<2d} | Train Loss: {:.4f}{}".format(
                str(self.idx), len(self.train_loader), loss, out_str
            )
        )

        g_k_1 = self._compute_gradiant(old_model=w_k_2, new_model=w_k_1)

        C_for_clip = self._compute_norm_l2_model_dict(g_k_1)**2
        # TODO : Adaptive C 
        beta_clip_fact = self._compute_beta(w_k_1, g_k_1, C_for_clip)
        g_all = self._compute_gradiant(old_model=origin_model, new_model=copy.deepcopy(model).to("cpu"))
        if self.args.debug:
            self.logging.info("Site-{:<5s} | before add noise model norm: {:.8f}".format(
                str(self.idx), 
                self._compute_norm_l2_model(model)
            ))
        
            self.logging.info("Site-{:<5s} | before clip gradiant norm: {:.8f}".format(
                str(self.idx), 
                self._compute_norm_l2_model_dict(g_all)
            ))
        
        if self.args.mode != 'no_dp' and sigma != None:
            # clip gradients and add noises
            sensitivity_params = self.clip_gradients(model, beta_clip_fact, origin_model, w_k_1.state_dict(), g_k_1)
            
            g_all = self._compute_gradiant(old_model=origin_model, new_model=copy.deepcopy(model).to("cpu"))
            if self.args.debug:
                self.logging.info("Site-{:<5s} | after clip gradiant norm: {:.8f}".format(
                    str(self.idx), 
                    self._compute_norm_l2_model_dict(g_all)
                ))
                self.logging.info("Site-{:<5s} | after clip model norm: {:.8f}".format(
                    str(self.idx), 
                    self._compute_norm_l2_model(model)
                ))

            if self.args.adp_noise:
                self.add_noise_per_param(model, sensitivity_params, sigma)
            # self.add_noise_per_param_on_g(model, sensitivity_params, sigma, g_all, origin_model)
            else:
                self.add_noise(model, sigma*C_for_clip)
            
            norm = 0
            for name, param in model.named_parameters():
                if name in sensitivity_params:
                    norm += torch.norm(param, 2).item()**2
            self.logging.info("Site-{:<5s} | after add noise norm: {:.8f}".format(str(self.idx), norm**0.5))
        return model.state_dict(), loss
    
    def test(self, model, mode):
        assert mode in ["val", "test"]
        test_loader = self.val_loader if mode == "val" else self.test_loader
        model.to(self.device)
        model.eval()
        loss_all = 0
        num_sample_test = 0

        segmentation = "UNet" in model.__class__.__name__
        test_acc = 0.0 if not segmentation else {}
        model_pred, label_gt, pred_prob = [], [], []
        for _, data in enumerate(test_loader):
            if self.args.data.startswith("prostate"):
                inp = data["Image"]
                target = data["Mask"]
                target = target.to(self.device)
            else:
                inp = data["Image"]
                target = data["Label"]
                target = target.to(self.device)

            inp = inp.to(self.device)
            output = model(inp)

            if self.args.data.startswith("prostate"):
                loss = self.loss_fun(output[:, 0, :, :], target)
            else:
                loss = self.loss_fun(output, target)

            loss_all += loss.item()

            if segmentation:
                if self.args.data.startswith("prostate"):
                    if len(test_acc.keys()) == 0:
                        test_acc["Dice"] = DiceLoss().dice_coef(output[:, 0, :, :], target).item()
                        test_acc["IoU"] = _eval_iou(output[:, 0, :, :], target).item()
                        test_acc["HD"] = 0.0
                    else:
                        test_acc["Dice"] += DiceLoss().dice_coef(output[:, 0, :, :], target).item()
                        test_acc["IoU"] += _eval_iou(output[:, 0, :, :], target).item()

                    for i_b in range(output.shape[0]):
                        hd = _eval_haus(output[i_b, 0, :, :], target[i_b]).item()
                        if hd > 0:
                            test_acc["HD"] += hd
                            num_sample_test += 1
                else:
                    if len(test_acc.keys()) == 0:
                        test_acc["Dice"] = DiceLoss().dice_coef(output, target).item()
                    else:
                        test_acc["Dice"] += DiceLoss().dice_coef(output, target).item()
            else:
                out_prob = torch.nn.functional.softmax(output, dim=1)
                model_pred.extend(out_prob.data.max(1)[1].view(-1).detach().cpu().numpy())
                pred_prob.extend(out_prob.data[:, 1].view(-1).detach().cpu().numpy())
                label_gt.extend(target.view(-1).detach().cpu().numpy())

        loss = loss_all / len(test_loader)
        # acc = test_acc/ len(data_loader) if segmentation else correct/total
        if segmentation:
            acc = {
                "Dice": test_acc["Dice"] / len(test_loader),
                "IoU": test_acc["IoU"] / len(test_loader),
                "HD": test_acc["HD"] / num_sample_test,
            }
        else:
            model_pred = np.asarray(model_pred)
            pred_prob = np.asarray(pred_prob)
            label_gt = np.asarray(label_gt)
            metric_res = metric_calc(label_gt, model_pred, pred_prob)
            acc = {
                "AUC": metric_res[1],
                "Acc": metric_res[2],
                "Sen": metric_res[3],
                "Spe": metric_res[4],
                "F1": metric_res[5],
            }
        model.to("cpu")
        return loss, acc

    
    def clip_gradients(self, model, beta, origin_model, model_weights, gradient):
        sensitivity_params = {}
        # get dict of model
        origin_model_dict = origin_model.state_dict()

        # each param in model.parameters(), clip_m = beta/2 * |model_weights - gradient|, param = min(max(param, param_orgin_model - clip_m), param_orgin_model + clip_m)
        for name, param in model.named_parameters():
            if name in model_weights and name in gradient:
                # clip_m = beta/2 * |model_weights - gradient|
                clip_m = beta / 2 * torch.norm(model_weights[name] - gradient[name], 2)**0.5
                sensitivity_params[name] = 2 * clip_m
                distance = torch.norm(param - origin_model_dict[name], 2)**0.5
                param.data = origin_model_dict[name] + (clip_m / distance if clip_m < distance else 1) * (param.data - origin_model_dict[name])

        return sensitivity_params

    def add_noise(self, model, sigma):
        for p in model.parameters():
            # add normal noise with std = sigma
            p.data += torch.normal(mean=0, std=sigma, size=p.size()).to(p.device)

        return model
    
    def add_noise_per_param(self, model, sensitivity_params, sigma):
        # get number of parameters
        params_num = len(model.state_dict())
        
        for name, param in model.named_parameters():
            if name in sensitivity_params:
                sigma_m = params_num**0.5 * sigma * sensitivity_params[name]
                noise = torch.normal(mean=0, std=sigma_m, size=param.size()).to(param.device)
                param.data += noise

        return model
    
    def add_noise_per_param_on_g(self, model, sensitivity_params, sigma, g_all, origin_model):
        # get number of parameters
        params_num = len(model.state_dict())
        
        for name in g_all.keys():
            if name in sensitivity_params:
                sigma_m = params_num**0.5 * sigma * sensitivity_params[name]
                noise = torch.normal(mean=0, std=sigma_m, size=g_all[name].size()).to("cpu")
                g_all[name].data += noise

        model_dict = origin_model.state_dict()
        for name in model_dict.keys():
            if name in g_all.keys():
                model_dict[name] += g_all[name]

        model.load_state_dict(model_dict)

        return model

    def test(self, model, testset):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in testset:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                log_probs = model(images)
                test_loss += self.loss_func(log_probs, labels).item()
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(labels.data.view_as(y_pred)).sum()
        test_loss /= len(testset.dataset)
        accuracy = 100. * correct / len(testset.dataset)
        return accuracy, test_loss
    
    def loss_func(self, log_probs, labels):
        return self.loss_fun(log_probs, labels)
    
    def get_grads(self, model):
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        return torch.cat(grads)
    
    def get_params(self, model):
        params = []
        for p in model.parameters():
            params.append(p.view(-1))
        return torch.cat(params)
    
    def _compute_gradiant(self, old_model, new_model):
        old_param = old_model.to("cpu").state_dict()
        new_param = new_model.to("cpu").state_dict()
        gradients = {}
        for name in old_param.keys():
            gradients[name] = new_param[name] - old_param[name]
        return gradients
    
    def _compute_beta(self, model, gradient, C):
        beta_clip_fact = 0
        model_weights = model.to("cpu").state_dict()
        for name in model_weights.keys():
            param_diff = model_weights[name] - gradient[name]  # w_k_1 - g_k_1
            beta_clip_fact += torch.norm(param_diff, 2).item()**2
        beta_clip_fact = C / beta_clip_fact  # Taking the square root to get the L2 norm
        self.logging.info(f"beta_clip_fact: {beta_clip_fact}")
        return beta_clip_fact
    
    def _compute_norm_l2_model(self, model):
        norm = 0
        for _, param in model.named_parameters():
            norm += torch.norm(param, 2).item()**2
        return norm**0.5
    
    def _compute_norm_l2_model_dict(self, model_dict):
        norm = 0
        for _, param in model_dict.items():
            norm += torch.norm(param, 2).item()**2
        return norm**0.5
    
