import torch
import os
import torch.nn as nn
import numpy as np
import torchvision

class RegressionModel(nn.Module):
    def __init__(self,classifier, init_mask, init_pattern):
        self._EPSILON = 1e-7
        super(RegressionModel, self).__init__()
        ### Here we initialize mask and pattern which will be trained
        ### pattern is the fix trigger to be learned
        self.mask_tanh = nn.Parameter(torch.tensor(init_mask),requires_grad=False)
        self.mask_tanh[:] = 0.1
        self.pattern_tanh = nn.Parameter(torch.tensor(init_pattern),requires_grad=True)
        ### we train pattern and mask similar to adversarial learning (classifier is fixed)
        for p in classifier.parameters():
            if p.requires_grad:
                p.requires_grad = False
        self.classifier = classifier.eval()

    def forward(self, x,padding_mask):
        # Write the dimensions
        mask = self.get_raw_mask()
        pattern = self.get_raw_pattern()
        x = (1 - mask) * x + mask * pattern
        return self.classifier(x,padding_mask, None,None)

    def get_raw_mask(self): # for offseting to make values [0,1]
        mask = nn.Tanh()(self.mask_tanh)
        return mask / (2 + self._EPSILON) + 0.5

    def get_raw_pattern(self):
        return self.pattern_tanh

    def get_raw_pattern_old(self):
        pattern = nn.Tanh()(self.pattern_tanh)
        return pattern / (2 + self._EPSILON) + 0.5






def outlier_detection(l1_norm_list, idx_mapping):
    print("-" * 30)
    print("Determining whether model is backdoor")

    consistency_constant = 1.4826
    median = torch.median(l1_norm_list)
    mad = consistency_constant * torch.median(torch.abs(l1_norm_list - median))
    min_mad = torch.abs(torch.min(l1_norm_list) - median) / mad

    print("Median: {}, MAD: {}".format(median, mad))
    print("Anomaly index: {}".format(min_mad))

    if min_mad < 2:
        print("Not a backdoor model")
    else:
        print("This is a backdoor model")

    flag_list = []
    for y_label in idx_mapping:
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if torch.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print(
        "Flagged label list: {}".format(",".join(["{}: {}".format(y_label, l_norm) for y_label, l_norm in flag_list]))
    )


def main(args,classifier,loader):

    # init_mask = np.random.randn(1, opt.input_height, opt.input_width).astype(np.float32)
    # init_pattern = np.random.randn(opt.input_channel, opt.input_height, opt.input_width).astype(np.float32)
    shape = (args.seq_len, args.enc_in)
    ##### burasi net degil seq_len ne T mi yoksa patchler mi? niye 2 dim var T x N mi ????
    init_mask = np.ones((shape[0],1)).astype(np.float32)
    init_pattern = np.ones(shape).astype(np.float32)

    for test in range(5):
        masks = []
        idx_mapping = {}

        for target_label in range(args.num_class):
            print("----------------- Analyzing label: {} -----------------".format(target_label))
            #args.target_label = target_label
            recorder, opt = train(args,classifier,loader, init_mask, init_pattern, target_label)

            mask = recorder.mask_best
            masks.append(mask)
            idx_mapping[target_label] = len(masks) - 1

        ## mask are None
        l1_norm_list = torch.stack([torch.sum(torch.abs(m)) for m in masks])
        print("{} labels found".format(len(l1_norm_list)))
        print("Norm values: {}".format(l1_norm_list))
        outlier_detection(l1_norm_list, idx_mapping)



class Recorder:
    def __init__(self, opt):
        super().__init__()

        # Best optimization results
        self.mask_best = None
        self.pattern_best = None
        self.reg_best = float("inf")

        # Logs and counters for adjusting balance cost
        self.logs = []
        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False

        # Counter for early stop
        self.early_stop_counter = 0
        self.early_stop_reg_best = self.reg_best

        # Cost
        self.cost = 0
        self.cost_multiplier = 2.0
        self.cost_multiplier_up = self.cost_multiplier
        self.cost_multiplier_down = self.cost_multiplier ** 1.5



def train(args,classifier, loader, init_mask, init_pattern,target_label):
    # Load the model

    # Build regression model
    regression_model = RegressionModel(classifier, init_mask, init_pattern).to(args.device)

    # Set optimizer
    optimizerR = torch.optim.Adam(regression_model.parameters(), lr=0.001)

    # Set recorder (for recording best result)
    recorder = Recorder(args)

    for epoch in range(50):
        train_step(regression_model, optimizerR, loader,recorder, epoch,args,target_label)

    # Save result to dir
    return recorder, args


def train_step(regression_model, optimizerR, dataloader, recorder, epoch, opt,target_label):
    print("Epoch {} - Label: {}".format(epoch, target_label))
    # Set losses
    atk_succ_threshold = 98.0
    cross_entropy = nn.CrossEntropyLoss()
    total_pred = 0
    true_pred = 0

    # Record loss for all mini-batches
    loss_ce_list = []
    loss_reg_list = []
    loss_list = []
    loss_acc_list = []

    # Set inner early stop flag
    inner_early_stop_flag = False
    for batch_idx, (inputs, labels,padding_mask) in enumerate(dataloader):
        # Forwarding and update model
        optimizerR.zero_grad()

        inputs = inputs.to(opt.device)
        padding_mask = padding_mask.to(opt.device)
        sample_num = inputs.shape[0]
        total_pred += sample_num
        target_labels = torch.ones((sample_num), dtype=torch.int64).to(opt.device) * target_label
        predictions = regression_model(inputs,padding_mask)

        loss_ce = cross_entropy(predictions, target_labels)
        #loss_reg = torch.norm(regression_model.get_raw_mask(), 1)
        loss_reg = torch.zeros_like(loss_ce)
        #total_loss = loss_ce + recorder.cost * loss_reg
        print('CE', loss_ce)
        total_loss = loss_ce
        total_loss.backward()
        optimizerR.step()

        # Record minibatch information to list
        minibatch_accuracy = torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach() * 100.0 / sample_num
        loss_ce_list.append(loss_ce.detach())
        loss_reg_list.append(loss_reg.detach())
        #loss_reg_list.append(0)
        loss_list.append(total_loss.detach())
        loss_acc_list.append(minibatch_accuracy)

        true_pred += torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach()
    loss_ce_list = torch.stack(loss_ce_list)
    loss_reg_list = torch.stack(loss_reg_list)
    loss_list = torch.stack(loss_list)
    loss_acc_list = torch.stack(loss_acc_list)

    avg_loss_ce = torch.mean(loss_ce_list)
    avg_loss_reg = torch.mean(loss_reg_list)
    avg_loss = torch.mean(loss_list)
    avg_loss_acc = torch.mean(loss_acc_list)
    print('avg Values:',avg_loss_reg,avg_loss_acc)

    # Check to save best mask or not
    if avg_loss_acc >= atk_succ_threshold and avg_loss_reg < recorder.reg_best:
        recorder.mask_best = regression_model.get_raw_mask().detach()
        recorder.pattern_best = regression_model.get_raw_pattern().detach()
        recorder.reg_best = avg_loss_reg
        print(" Updated !!!")

    # Show information
    print(
        "  Result: Accuracy: {:.3f} | Cross Entropy Loss: {:.6f} | Reg Loss: {:.6f} | Reg best: {:.6f}".format(
            true_pred * 100.0 / total_pred, avg_loss_ce, avg_loss_reg, recorder.reg_best
        )
    )
    return
