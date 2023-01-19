# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/11/25 19:00
import random
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable
from Utils import Tools, LossFunction, Scripts

def train_on_batch(opt, epochs, train_iter, G, D, device, subject=1, source=0, fold_num=0, lr_jitter=False):
    # cross entropy loss and optimizer
    ce_loss1 = nn.CrossEntropyLoss()
    ce_loss1 = ce_loss1.to(device)
    ce_loss2 = nn.CrossEntropyLoss()
    ce_loss2 = ce_loss2.to(device)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)
    d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=epochs * len(train_iter), eta_min=5e-6)
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=epochs * len(train_iter), eta_min=5e-6)
    lecam_ema = LossFunction.LeCamEMA(decay=0.99, start_iter=opt.start)

    Tools.CLE_DIR_CONENT(f'../Generation/TEGAN/{opt.dataset}')

    epoch_list = []
    # evaluation metrics
    d_loss_list, g_loss_list = [], []
    real_acc_list, fake_acc_list, d_acc_list = [], [], []

    G.train()
    D.train()
    for epoch in range(epochs):
        sum_d_loss, sum_g_loss = 0.0, 0.0
        sum_real_acc, sum_fake_acc, sum_d_acc = 0.0, 0.0, 0.0
        len_iter = len(train_iter)
        for i, (eeg, labels) in enumerate(train_iter):
            # obtain the number of a batch of eeg
            num_eeg = eeg.shape[0]
            eeg = eeg.float()
            real_eeg = Variable(eeg).to(device)

            # Configure real conditional labels
            labels = labels.squeeze(-1)
            labels = labels.long().to(device)

            # ==============train discriminator================
            d_optimizer.zero_grad()
            # compute loss of real_eeg
            real_prob, real_aux = D(real_eeg)
            aux_loss_real = ce_loss1(real_aux, labels)

            # compute loss of fake_eeg
            z = real_eeg[:, :, :, :round(opt.ws * opt.Fs)]
            fake_eeg, fake_label = G(z)
            fake_prob, fake_aux = D(fake_eeg.detach())
            aux_loss_fake = ce_loss1(fake_aux, labels)

            # backpropagation and optimize
            d_loss_adv = LossFunction.d_hinge(real_prob, fake_prob) / 2

            d_loss_aux = (aux_loss_real + aux_loss_fake) / 2
            d_loss = 0.5 * d_loss_adv + 0.5 * d_loss_aux

            lecam_ema.update(torch.mean(real_prob).item(), 'D_real', epoch)
            lecam_ema.update(torch.mean(fake_prob).item(), 'D_fake', epoch)
            if epoch > lecam_ema.start_itr:
                lecam_loss = LossFunction.lecam_reg(real_prob, fake_prob, lecam_ema)
            else:
                lecam_loss = torch.tensor(0., device=device)

            d_loss += 0.3 * lecam_loss
            d_loss.backward()
            d_optimizer.step()

            if lr_jitter:
                d_scheduler.step()

            # =============train generator=============
            g_optimizer.zero_grad()

            # compute loss of fake_eeg
            validity, pred_label = D(fake_eeg)
            ce_loss_gen = 0.5 * ce_loss1(pred_label, labels)

            adv_loss_gen = LossFunction.g_hinge(validity)

            g_loss = 0.5 * adv_loss_gen + 0.5 * ce_loss_gen

            # backpropagation and optimize
            g_loss.backward()
            g_optimizer.step()
            if lr_jitter:
                g_scheduler.step()

            # Calculate discriminator accuracy
            real_aux = F.softmax(real_aux, dim=-1)
            fake_aux = F.softmax(fake_aux, dim=-1)
            pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
            gt = np.concatenate([labels.data.cpu().numpy(), labels.data.cpu().numpy()], axis=0)
            d_real_acc = np.mean(np.argmax(real_aux.data.cpu().numpy(), axis=1) == labels.data.cpu().numpy())
            d_fake_acc = np.mean(np.argmax(fake_aux.data.cpu().numpy(), axis=1) == labels.data.cpu().numpy())
            d_acc = np.mean(np.argmax(pred, axis=1) == gt)

            # accumulate the loss and scores
            sum_d_loss += d_loss.item()
            sum_g_loss += g_loss.item()
            sum_real_acc += d_real_acc
            sum_fake_acc += d_fake_acc
            sum_d_acc += d_acc

            if epoch % 50 == 0:
                stochastic_index = random.randint(0, num_eeg - 1)
                # after each iteration,plot real EEG
                real_eegs = real_eeg.cpu().data
                plt_eeg = real_eegs[stochastic_index]
                labels = labels.cpu().data
                plt_label = labels[stochastic_index]
                plt_eeg = plt_eeg.reshape(opt.Nc, round(opt.factor * opt.Fs * opt.ws))
                plt_label.squeeze(-1)
                Tools.plot_TimeEEG(subject, plt_eeg, plt_label, epoch, model_name="TEGAN", dataset=opt.dataset,
                                   real_or_fake='real')
                Tools.plot_TimeFreqEEG2CH(subject, plt_eeg, plt_label, epoch, model_name="TEGAN", dataset=opt.dataset,
                                          real_or_fake='real')

                # after each iteration,plot fake EEG
                fake_eegs = fake_eeg.cpu().data
                plt_eeg = fake_eegs[stochastic_index]
                labels = labels.cpu().data
                plt_label = labels[stochastic_index]
                plt_eeg = plt_eeg.reshape(opt.Nc, round(opt.factor * opt.Fs * opt.ws))
                plt_label.squeeze(-1)
                Tools.plot_TimeEEG(subject, plt_eeg, plt_label, epoch, model_name="TEGAN", dataset=opt.dataset,
                                   real_or_fake='fake')
                Tools.plot_TimeFreqEEG2CH(subject, plt_eeg, plt_label, epoch, model_name="TEGAN", dataset=opt.dataset,
                                          real_or_fake='fake')

        epoch_list.append(epoch + 1)
        d_loss_list.append(sum_d_loss / len_iter)
        g_loss_list.append(sum_g_loss / len_iter)
        real_acc_list.append(sum_real_acc / len_iter)
        fake_acc_list.append(sum_fake_acc / len_iter)
        d_acc_list.append(sum_d_acc / len_iter)

        # print the evaluation metrics for training process
        print(f'epoch{epoch + 1} d_loss={sum_d_loss / len_iter:.3f}, g_loss={sum_g_loss / len_iter:.3f}, '
              f'real_acc:{sum_real_acc / len_iter:.3f}, fake_acc:{sum_fake_acc / len_iter:.3f}, '
              f'cls_acc:{sum_d_acc / len_iter:.3f}')

    if source == 0:
        torch.save(G.state_dict(), f'../Pretrain/{opt.dataset}/{opt.ws}S-{opt.ws*opt.factor}S/'
                                   f'Source/TEGAN_FC_Gs_S{subject}.pth')
        torch.save(D.state_dict(), f'../Pretrain/{opt.dataset}/{opt.ws}S-{opt.ws*opt.factor}S/'
                                   f'Source/TEGAN_FC_Ds_S{subject}.pth')

    else:
        torch.save(G.state_dict(), f'../Pretrain/{opt.dataset}/{opt.ws}S-{opt.ws*opt.factor}S/'
                                   f'Target/R{opt.ratio}/F{fold_num}/TEGAN_FC_Gt_S{subject}.pth')


    plt.plot(epoch_list, d_loss_list)
    plt.plot(epoch_list, g_loss_list)
    plt.plot(epoch_list, real_acc_list)
    plt.plot(epoch_list, fake_acc_list)
    plt.plot(epoch_list, d_acc_list)
    plt.xlabel('epoch')
    plt.title('Training process for TEGAN')
    plt.legend(['d_loss', 'g_loss', 'real_acc', 'fake_acc', 'd_acc'], loc='upper right')
    plt.savefig(f'../Figure/Loss_Curve/{opt.dataset}/TEGAN_S{subject}.png')
    plt.show()