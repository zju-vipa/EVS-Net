
#from utility import *
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torch import autograd
import math

from models.modeling.deeplab import *

from utils import *

#import segmentation_models_pytorch as smp
def get_generator_model(self):
    #opt.num_class, opt.backbone, opt.out_stride, opt.sync_bn, opt.freeze_bn 
    

    model = DeepLab(
            num_classes = self.opt.num_class,
            backbone = self.opt.backbone,
            output_stride = self.opt.out_stride,
            sync_bn = self.opt.sync_bn,
            freeze_bn = self.opt.freeze_bn
            )
    
    return model

def get_generator_optimizer(self,model):
    
    model_params = [
            {'params': model.get_1x_lr_params(), 'lr': self.opt.lr},
            {'params': model.get_10x_lr_params(), 'lr': self.opt.lr * 10}
            ]
    optimizer = torch.optim.SGD(model_params,lr=self.opt.lr,momentum=self.opt.momentum,weight_decay=self.opt.weight_decay,nesterov=True)
    init_lr, end_lr, epochs, offset = self.opt.lr, self.opt.min_lr, self.opt.epochs, self.opt.epoch_interval

    k = epochs // offset
    gamma = math.exp(math.log(end_lr / init_lr)/k)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=offset, gamma=gamma)
    return optimizer,scheduler

def get_ralamb_optimizer(self, model):
    ralam = Ralamb(model.parameters())
    optimizer = Lookahead(ralam,0.5,6)
    return optimizer, None

def get_adam_optimizer(self, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return optimizer, None

class Discriminator(nn.Module):
    def __init__(self, opt, channels=9, c_dim=1, n_strided=6):
        super(Discriminator, self).__init__()
        img_size = opt.crop_size
        def discriminator_block(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1), nn.LeakyReLU(0.01)]
            return layers

        layers = discriminator_block(channels, 16)
        curr_dim = 16
        for _ in range(n_strided - 1):
            layers.extend(discriminator_block(curr_dim, curr_dim * 2))
            curr_dim *= 2

        self.model = nn.Sequential(*layers)

        # Output 1: PatchGAN
        self.out1 = nn.Sequential(nn.Conv2d(curr_dim, 1, 3, padding=1, bias=False))
        # Output 2: Class prediction
        #kernel_size = img_size // 2 ** n_strided
        #self.out2 = nn.Conv2d(curr_dim, c_dim, kernel_size, bias=False)

    def forward(self, img):
        feature_repr = self.model(img)
        out_adv = self.out1(feature_repr)
        out_adv = torch.sigmoid(out_adv)

        #out_cls = self.out2(feature_repr)
        return out_adv#,None #out_cls.view(out_cls.size(0), -1).squeeze()



def get_discriminator_model(self,**kwargs):
    # input_shape + mask_shape
    if "channels" in kwargs.keys():
        model = Discriminator(self.opt,channels=kwargs['channels'])
    else:
        model = Discriminator(self.opt)
    
    model.apply(weight_init)
    
    #model = _resDiscriminator128(nIn=9, selfAtt=True)
    #model.apply(weight_init)

    return model
    
def get_discriminator_optimizer(self,model):
    params = split_weights(model)
    optimizer = torch.optim.SGD(params,lr=self.opt.lr,momentum=self.opt.momentum,weight_decay=self.opt.weight_decay,nesterov=True)
    init_lr, end_lr, epochs, offset = self.opt.lr, self.opt.min_lr, self.opt.epochs, self.opt.epoch_interval

    k = epochs // offset
    gamma = math.exp(math.log(end_lr / init_lr)/k)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=offset, gamma=gamma)
    return optimizer,scheduler

def compute_gradient_penalty(D, real_samples, fake_samples,device=torch.device(0)):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha =  torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1))).cuda(device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    
    fake = Variable(torch.FloatTensor(np.ones(d_interpolates.shape)).cuda(device=device), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    #print("gradient penalty --> ", gradient_penalty)
    return gradient_penalty


##############################  model factory  #####################################

class ModelFactory(object):
    def __init__(self, opt):
        self.opt = opt

    def get_default_model(self):
        assert(hasattr(self.opt,'label_num'))
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(2048,self.opt.label_num)
        init.xavier_normal_(model.fc.weight.data,gain=0.02)
        init.constant_(model.fc.bias.data,0.0)
        return model
    
    def get_default_optimizer(self,model):
        params = split_weights(model)
        optimizer = torch.optim.SGD(params,lr=self.opt.lr,momentum=self.opt.momentum,weight_decay=self.opt.weight_decay,nesterov=True)
        init_lr, end_lr, epochs, offset = self.opt.lr, self.opt.min_lr, self.opt.epochs, self.opt.epoch_interval

        k = epochs // offset
        gamma = math.exp(math.log(end_lr / init_lr)/k)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=offset, gamma=gamma)
        return optimizer,scheduler

    def model_span(self,model,requires_grad=True):
        for param in model.parameters():
            param.requires_grad = requires_grad
        return model

    def __call__(self, default=True,**kwargs):

        model = self.get_default_model(self,**kwargs)
        optimizer, scheduler = self.get_default_optimizer(self,model)

        return model, optimizer, scheduler

    def register_hook_model(self, func):
        self.get_default_model = func
    def register_hook_optimizer(self, func):
        self.get_default_optimizer = func

        
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
        
class WeightClipper(object):
    def __init__(self,frequency=1):
        self.freg = frequency
    def __call__(self, module):
        if hasattr(module,'weight'):
            w = module.weight.data
            w = w.clamp(-0.1,0.1)

###########################################################################################
# def dense_crf(output):
#     B,C,H,W = output.shape
#     #pdb.set_trace()

#     output = output.cpu().numpy() 
#     out = []
#     for img in output:
#         tmp = np.concatenate([1-img, img])
#         U = crfutils.unary_from_softmax(tmp)
#         U = np.ascontiguousarray(U)
#         d = dcrf.DenseCRF2D(W,H,C*2)

#         d.setUnaryEnergy(U)
#         d.addPairwiseGaussian(sxy=5, compat=10)

#         Q = d.inference(25)
#         map_soln = np.argmax(Q, axis=0).reshape(1,H,W)

#         #Q = np.array(Q).reshape((C*2, H, W))
#         out.append(map_soln)
#     out = torch.from_numpy(np.asarray(out))
#     return out


if __name__ == "__main__":
    import pdb

    opt = parse_opts()
    x = torch.FloatTensor(32,1,64,64).cuda(1)
    x = torch.sigmoid(x)
    #pdb.set_trace()
    z = dense_crf(x)

