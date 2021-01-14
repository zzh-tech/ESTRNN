import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.nn.modules.loss import _Loss


def MSE(para):
    """
    L2 loss
    """
    return nn.MSELoss()


def L1(para):
    """
    L1 loss
    """
    return nn.L1Loss()


class L1GradientLoss(_Loss):
    """
    Gradient loss
    """

    def __init__(self, para):
        super(L1GradientLoss, self).__init__()
        self.get_grad = Gradient()
        self.L1 = nn.L1Loss()

    def forward(self, x, y):
        grad_x = self.get_grad(x)
        grad_y = self.get_grad(y)
        loss = self.L1(grad_x, grad_y)
        return loss


class Gradient(nn.Module):
    def __init__(self):
        super(Gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x


class L1_Charbonnier_loss(_Loss):
    """
    L1 Charbonnierloss
    """

    def __init__(self, para):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps * self.eps)
        loss = torch.mean(error)
        return loss


class L1_Charbonnier_loss_color(_Loss):
    """
    L1 Charbonnierloss color
    """

    def __init__(self, para):
        super(L1_Charbonnier_loss_color, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        diff_sq = diff * diff
        # print(diff_sq.shape)
        diff_sq_color = torch.mean(diff_sq, 1, True)
        # print(diff_sq_color.shape)
        error = torch.sqrt(diff_sq_color + self.eps * self.eps)
        loss = torch.mean(error)
        return loss


def Perceptual(para):
    return PerceptualLoss(loss=nn.L1Loss())


class PerceptualLoss:
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def __call__(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class GAN(nn.Module):
    """
    GAN loss
    """

    def __init__(self, para):
        super(GAN, self).__init__()
        self.device = torch.device('cpu' if para.cpu else 'cuda')
        self.D = Discriminator().to(self.device)
        self.criterion = nn.BCELoss().to(self.device)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=para.lr)
        self.real_label = 1
        self.fake_label = 0

    def forward(self, x, y, valid_flag=False):
        self.D.zero_grad()
        b = y.size(0)
        label = torch.full((b,), self.real_label, device=self.device)
        if not valid_flag:
            ############################################
            # update D network: maximize log(D(y) + log(1-D(G(x))))
            # train with all-real batch
            output = self.D(y).view(-1)  # forward pass of real batch through D
            errD_real = self.criterion(output, label)  # calculate loss on all-real batch
            errD_real.backward()  # calculate gradients for D in backward pass
            ## D_y = output.mean().item()
            # train with all-fake batch
            label.fill_(self.fake_label)
            output = self.D(x.detach()).view(-1)  # classify all fake batch with D
            errD_fake = self.criterion(output, label)  # calculate D's loss on the all-fake batch
            errD_fake.backward()  # calculate gradients for all-fake batch
            ## D_G_x1 = output.mean().item()
            # add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # update D
            self.D_optimizer.step()
        ############################################
        # generate loss for G network: maximize log(D(G(x)))
        label.fill_(self.real_label)  # fake labels are all real for generator cost
        # since we just updated D, perform another forward pass of all-fake batch through D
        output = self.D(x).view(-1)
        errG = self.criterion(output, label)  # calculate G's loss on this output
        ## D_G_x2 = output.mean().item()

        return errG


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # x: (n,3,256,256)
        c = 3
        h, w = 256, 256
        n_feats = 8
        n_middle_blocks = 6
        self.start_module = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=n_feats, kernel_size=3, stride=1, padding=1),  # (n,8,256,256)
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            BasicBlock(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=2, padding=1),  # (n,8,128,128)
        )
        middle_module_list = []
        for i in range(n_middle_blocks):
            middle_module_list.append(
                BasicBlock(in_channels=n_feats * (2 ** i), out_channels=n_feats * (2 ** (i + 1)), kernel_size=3,
                           stride=1, padding=1))
            middle_module_list.append(
                BasicBlock(in_channels=n_feats * (2 ** (i + 1)), out_channels=n_feats * (2 ** (i + 1)), kernel_size=3,
                           stride=2, padding=1))
        self.middle_module = nn.Sequential(*middle_module_list)
        self.end_module = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (n,3,256,256)
        n, _, _, _ = x.shape
        h = self.start_module(x)  # (n,8,256,256)
        h = self.middle_module(h)  # (n,512,2,2)
        h = h.reshape(n, -1)  # (n,2048)
        out = self.end_module(h)  # (n,1)
        return out


def loss_parse(loss_str):
    """
    parse loss parameters
    """
    ratios = []
    losses = []
    str_temp = loss_str.split('|')
    for item in str_temp:
        substr_temp = item.split('*')
        ratios.append(float(substr_temp[0]))
        losses.append(substr_temp[1])
    return ratios, losses


class Loss(nn.Module):
    """
    Training loss
    """

    def __init__(self, para):
        super(Loss, self).__init__()
        ratios, losses = loss_parse(para.loss)
        self.losses_name = losses
        self.ratios = ratios
        self.losses = []
        for loss in losses:
            loss_fn = eval('{}(para)'.format(loss))
            self.losses.append(loss_fn)

    def _forward_single(self, x, y, valid_flag=False):
        if len(x.shape) == 5:
            b, n, c, h, w = x.shape
            x = x.reshape(b * n, c, h, w)
            y = y.reshape(b * n, c, h, w)
        losses = {}
        loss_all = None
        for i in range(len(self.losses)):
            if valid_flag == True and self.losses_name[i] == 'GAN':
                loss_sub = self.ratios[i] * self.losses[i](x, y, valid_flag)
            else:
                loss_sub = self.ratios[i] * self.losses[i](x, y)
            losses[self.losses_name[i]] = loss_sub
            if loss_all == None:
                loss_all = loss_sub
            else:
                loss_all += loss_sub
        losses['all'] = loss_all

        return losses

    def _forward_list(self, x, y, valid_flag=False):
        assert len(x) == len(y)
        scales = len(x)
        losses = None
        for i in range(scales):
            temp_losses = self._forward_single(x[i], y[i], valid_flag)
            if losses is None:
                losses = temp_losses
            else:
                for key in losses.keys():
                    losses[key] += temp_losses[key]
        return losses

    def forward(self, x, y, valid_flag=False):
        if isinstance(x, (list, tuple)):
            B, N, C, H, W = y.shape
            _y = []
            _y.append(y)
            y = y.reshape(B, N * C, H, W)
            _y.append(
                F.interpolate(y, size=(H // 2, W // 2), mode='bilinear', align_corners=False).reshape(B, N, C, H // 2,
                                                                                                      W // 2))
            _y.append(
                F.interpolate(y, size=(H // 4, W // 4), mode='bilinear', align_corners=False).reshape(B, N, C, H // 4,
                                                                                                      W // 4))
            return self._forward_list(x, _y, valid_flag)
        else:
            return self._forward_single(x, y, valid_flag)
