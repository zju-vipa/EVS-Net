from dataloader.dataset import *
from utils.utility import *
from utils.metric import *
from models.model import *
from loss.seg_loss import *
from torch.autograd import Variable
from scipy import ndimage
import copy
import cv2


def get_training_data(opt):
    
    try:
        input_img, input_mask = next(opt.data_iter)
        real_input, real_mask = next(opt.real_iter)
        
        if input_img.size(0) < opt.batch_size:
            input_img, input_mask = next(opt.data_iter)
        if real_input.size(0) < opt.batch_size:
            real_input, real_mask = next(opt.real_iter)

    except StopIteration:
        opt.data_iter = iter(opt.train_loader)
        opt.real_iter = iter(opt.real_loader)
        input_img, input_mask = next(opt.data_iter)
        real_input, real_mask = next(opt.real_iter)

    input_img, input_mask = input_img.cuda(opt.gpu), input_mask.cuda(opt.gpu)
    real_input, real_mask = real_input.cuda(opt.gpu), real_mask.cuda(opt.gpu)

    return input_img, input_mask, real_input, real_mask

def get_shot_data(opt):
    try:
        input_img, input_mask = next(opt.shot_iter)
        
        if input_img.size(0) < opt.batch_size:
            input_img, input_mask = next(opt.shot_iter)

    except StopIteration:
        opt.shot_iter = iter(opt.shot_loader)
        input_img, input_mask = next(opt.shot_iter)

    input_img, input_mask = input_img.cuda(opt.gpu), input_mask.cuda(opt.gpu)
    

    return input_img, input_mask


def get_discriminator_input(opt,input_img, fake_mask, real_input, real_mask, flag=True):
    
    if flag:
        temp_fake = fake_mask * input_img
        temp_real = real_mask * real_input

        fake = torch.cat([input_img,  temp_fake, fake_mask, fake_mask, fake_mask], dim=1)
        real = torch.cat([real_input, temp_real, real_mask, real_mask, real_mask], dim=1)

        return real, fake
    else:
       
        temp_mask = 1 - fake_mask
        temp_fake = temp_mask*input_img
        temp_real = (1-real_mask)*real_input

        fake = torch.cat([input_img, temp_fake, temp_mask, temp_mask, temp_mask], dim=1)
        real = torch.cat([real_input, temp_real, 1-real_mask, 1-real_mask, 1-real_mask], dim=1)
        
        return real, fake

def get_close_opening(mask, flag=True, kernel_size=11):
    if flag:
        mask = F.max_pool2d(mask, kernel_size=kernel_size,stride=1, padding=kernel_size//2  )
        mask = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=kernel_size//2 )
        mask = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=kernel_size//2 )
        mask = F.max_pool2d(mask, kernel_size=kernel_size,stride=1, padding=kernel_size//2  )
        
    else:
        mask = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=kernel_size//2 )
        mask = F.max_pool2d(mask, kernel_size=kernel_size,stride=1, padding=kernel_size//2  )
        mask = F.max_pool2d(mask, kernel_size=kernel_size,stride=1, padding=kernel_size//2  )
        mask = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=kernel_size//2 )
    return mask

def get_erosion_dilation(input_img, input_mask, flag=True, kernel_size=25):
    if flag:
        temp_mask = - F.max_pool2d(-input_mask, kernel_size=kernel_size,stride=1, padding=kernel_size//2  )
        temp_fake = temp_mask * input_img
        return torch.cat([input_img, temp_fake, temp_mask, temp_mask, temp_mask], dim=1)
    else:
        temp_mask = F.max_pool2d(input_mask, kernel_size=kernel_size,stride=1, padding=kernel_size//2  )
        temp_fake = (1-temp_mask) * input_img
        return torch.cat([input_img, temp_fake, 1-temp_mask, 1-temp_mask, 1-temp_mask,], dim=1)
    
def discriminate_triplet(D, real, fake, pseudo, opt, flag="erosion"):
    
    fake_validity = D(fake)
    pseudo_validity = D(pseudo)
    d_fake, d_pseudo =  torch.mean(fake_validity), torch.mean(pseudo_validity)
    gradient_penalty = compute_gradient_penalty(D,pseudo.data,fake.data,device=opt.gpu)
    d_penalty = opt.lambda_gp * gradient_penalty
    d_fake.backward(opt.one, retain_graph=True)
    d_pseudo.backward(opt.mone, retain_graph=True)
    d_penalty.backward()
    d_loss =  d_fake - d_pseudo + d_penalty 
    
    if flag == "erosion":
        opt.writer.update_loss( 
                                d_erosion_fake    = d_fake.data.cpu().item(),\
                                d_erosion_pseudo  = d_pseudo.data.cpu().item(),\
                                d_erosion_penalty = gradient_penalty.data.cpu().item())


    elif flag == "dilation":
        opt.writer.update_loss( 
                                d_dilation_fake    = d_fake.data.cpu().item(),\
                                d_dilation_pseudo  = d_pseudo.data.cpu().item(),\
                                d_dilation_penalty = gradient_penalty.data.cpu().item())
        
    return d_loss
    

def train(opt, epoch):

    opt.D_f.train(True) 
    opt.D_b.train(True)
    opt.G.train(True)
    opt.writer.reset()
    

    opt.optim_D_f.zero_grad()
    opt.optim_D_b.zero_grad()
    input_imgs, input_mask, real_input,real_mask = get_training_data(opt)
    # k = int(np.random.choice(range(3,7,2),1)) 
    k = 5
    # kp = int(np.random.choice(range(3,19,2),1)) 
    kp = 11

    ## Train Discriminator
    fake_mask = opt.G(input_imgs)
    
    real, fake = get_discriminator_input(opt, input_imgs, fake_mask.detach(), real_input, real_mask,flag=True)
    pseudo = get_erosion_dilation(real_input, real_mask,flag=True, kernel_size= kp )
    loss_erosion = discriminate_triplet(opt.D_f, real, fake, pseudo, opt, flag="erosion")
        
    real_, fake_ = get_discriminator_input(opt, input_imgs, fake_mask, real_input, real_mask,flag=False)       
    pseudo_ = get_erosion_dilation(real_input, real_mask,flag=False, kernel_size=kp)
    loss_dilation = discriminate_triplet(opt.D_b, real_, fake_, pseudo_, opt, flag="dilation")

    # discriminative loss
    loss = loss_erosion + loss_dilation

    opt.optim_D_f.step()
    opt.optim_D_b.step()

    opt.D_f.apply(opt.clipper)
    opt.D_b.apply(opt.clipper)

    print("[Discriminator Step --> Epoch %d/%d] [D erosion loss: %.4f] [D dilation loss: %.4f] "%(opt.epochs, epoch,loss_erosion.item(),loss_dilation.item()))

    if epoch % 5 == 0:
        shot_num = 5
        for idx in range(shot_num):
        
            few_imgs, few_mask = get_shot_data(opt)
            opt.optim_G.zero_grad()
            few_imgs,few_mask = few_imgs.cuda(opt.gpu), few_mask.cuda(opt.gpu)
            few_self = opt.self_op(few_imgs, flag=True)

            few_out = opt.G(few_imgs)
            few_op = opt.G(few_self)
            
            few_out_ero = - F.max_pool2d(-few_out, kernel_size=k,stride=1, padding=k//2  )
            few_out_dil = F.max_pool2d(few_out, kernel_size=k,stride=1, padding=k//2  )
            few_out_mask = few_out_dil - few_out_ero
            few_out_self = few_out * few_out_mask

            few_op_ero = - F.max_pool2d(-few_op, kernel_size=k,stride=1, padding=k//2  )
            few_op_dil = F.max_pool2d(few_op, kernel_size=k,stride=1, padding=k//2  )
            few_op_mask = few_op_dil - few_op_ero
            few_op_self = few_op * few_op_mask

            raw_loss = opt.dice_criterion(few_out, few_mask)
            self_loss = ((few_out_self - opt.self_op(few_op_self, flag=False))**2).mean()

            few_loss = raw_loss + self_loss

            few_loss.backward()
            opt.optim_G.step()
            opt.G.apply(opt.clipper)

        opt.optim_G.zero_grad()
        self_imgs = opt.self_op(input_imgs, flag=True)

        fake_mask = opt.G(input_imgs)
        self_mask = opt.G(self_imgs)

        _, fake = get_discriminator_input(opt, input_imgs, fake_mask, real_input, real_mask,flag=True)
        _, fake_ = get_discriminator_input(opt, input_imgs, fake_mask, real_input, real_mask,flag=False)


        fake_erosion_validity = opt.D_f(fake)
        fake_dilation_validity = opt.D_b(fake_)

        g_fake_erosion , g_fake_dilation = torch.mean(fake_erosion_validity), torch.mean(fake_dilation_validity)
        
        
        fake_mask_ero = - F.max_pool2d(-fake_mask, kernel_size=k,stride=1, padding=k//2  )
        fake_mask_dil = F.max_pool2d(fake_mask, kernel_size=k,stride=1, padding=k//2  )
        fake_mask_mask = fake_mask_dil - fake_mask_ero
        fake_mask_self = fake_mask * fake_mask_mask

        self_mask_ero = - F.max_pool2d(-self_mask, kernel_size=k,stride=1, padding=k//2  )
        self_mask_dil = F.max_pool2d(self_mask, kernel_size=k,stride=1, padding=k//2  )
        self_mask_mask = self_mask_dil - self_mask_ero
        self_mask_self = self_mask * self_mask_mask

        self_loss = ((fake_mask_self - opt.self_op(self_mask_self, flag=False))**2).mean()
        
        self_loss.backward(retain_graph=True)
        g_fake_erosion.backward(opt.mone, retain_graph=True)
        g_fake_dilation.backward(opt.mone,retain_graph=True)

        g_loss = - g_fake_erosion - g_fake_dilation + self_loss

        opt.optim_G.step()
        opt.G.apply(opt.clipper)

        opt.writer.update_loss(g_erosion_fake=g_fake_erosion.data.cpu().item(),g_dilation_fake=g_fake_dilation.item(),self_loss=self_loss.data.cpu().item())
        
        metric_data = opt.writer.evaluator(input_mask.data, opt.writer.refiner.bin(fake_mask))
        
        opt.writer.update_metric(**metric_data)

        print("[Generator Step --> Epoch %d/%d] [G loss: %.4f]"%(opt.epochs, epoch, g_loss.item()))

    opt.writer.dump_loss("Training Loss",epoch)
    if epoch % 5 == 0:
        opt.writer.dump_metric("Training Metric",epoch//5)


def validate(opt,epoch):
    opt.G.eval()
    opt.writer.reset()
    
    with torch.no_grad():
        for idx, (input_imgs, input_mask) in enumerate(opt.val_loader):
            
            input_imgs = input_imgs.cuda(opt.gpu)
            predicted = opt.G(input_imgs)
            refine = opt.writer.refiner.bin(predicted)
            opt.writer.update_metric(**opt.writer.evaluator(input_mask, refine))
            if idx < 5:
                opt.writer.add_images("val_label_%03d"%(idx), input_mask, epoch, False, False, False)
                opt.writer.add_images("val_predict_raw_%03d"%(idx), predicted, epoch, False,False,False)
                opt.writer.add_images("val_predict_full_%03d"%(idx), refine, epoch, False,False,False)
                input_mask_ = input_mask.cpu().numpy().repeat(3,axis=1)
                predicted_ = predicted.cpu().numpy().repeat(3,axis=1)
                refine_ = refine.cpu().numpy().repeat(3,axis=1)
                
                input_imgs = input_imgs.cpu().numpy()
                pre = predicted_
                ref = refine_
                
            
        metric_data = opt.writer.dump_metric("Val Metric", epoch//100)
        opt.logger.save_checkpoint(state_dict=opt.G.state_dict(), scores=metric_data, epoch=epoch)


def init_settings(opt):
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(1124)
    opt.gpu = torch.device(opt.device_id)
    opt.dice_criterion = DiceLoss().cuda(opt.gpu)
    opt.bce_criterion = nn.BCEWithLogitsLoss().cuda(opt.gpu)
    opt.self_op = SelfOperation(opt)

    opt = init_models(opt)
    opt = init_dataset(opt)
    opt = folder_init(opt)
    opt.writer = TensorWriter(opt)
    opt.logger = Reseroir(opt)

    opt.ones = torch.ones(opt.batch_size).cuda(opt.gpu)
    opt.zeros = torch.zeros(opt.batch_size).cuda(opt.gpu)

    opt.one = torch.tensor(1,dtype=torch.float).cuda(opt.gpu)
    opt.mone = -1.0 * opt.one

    return opt

def init_models(opt):
    # model init
    generator = ModelFactory(opt)
    generator.register_hook_model(get_generator_model)
    generator.register_hook_optimizer(get_ralamb_optimizer)

    opt.G, opt.optim_G, opt.sche_G = generator()
    opt.G.cuda(opt.gpu)

    discriminator_f = ModelFactory(opt)
    discriminator_f.register_hook_model(get_discriminator_model)
    discriminator_f.register_hook_optimizer(get_ralamb_optimizer)

    opt.D_f, opt.optim_D_f, opt.sche_D_f = discriminator_f()
    opt.D_f.cuda(opt.gpu)

    discriminator_b= ModelFactory(opt)
    discriminator_b.register_hook_model(get_discriminator_model)
    discriminator_b.register_hook_optimizer(get_ralamb_optimizer)

    opt.D_b, opt.optim_D_b, opt.sche_D_b = discriminator_b()
    opt.D_b.cuda(opt.gpu)

    opt.clipper = WeightClipper()

    return opt

def init_dataset(opt):
    

    folder1 = ''
    folder2 = ''
    folder3 = ''

    shotnum = 10
    opt.portrait_train = SegmentationDataset(opt, split="train",folder=folder1)
    opt.portrait_real = SegmentationDataset(opt, split="full", folder=folder2)
    opt.portrait_shot = OneShotDataset(opt,train_folder=folder1, target_folder=folder3,minnum=shotnum)
    opt.portrait_val = SegmentationDataset(opt, split="val", folder=folder1)

    opt.train_loader =  DataLoader(opt.portrait_train,collate_fn=segmentation_collate_fn, batch_size=opt.batch_size, shuffle=True, pin_memory=True,num_workers=4)
    opt.real_loader = DataLoader(opt.portrait_real,collate_fn=segmentation_collate_fn, batch_size=opt.batch_size, shuffle=True, pin_memory=True,num_workers=4)
    opt.shot_loader = DataLoader(opt.portrait_shot,collate_fn=segmentation_collate_fn, batch_size=opt.batch_size, shuffle=True, pin_memory=True,num_workers=4)
    opt.val_loader =  DataLoader(opt.portrait_val,collate_fn=segmentation_collate_fn, batch_size=opt.batch_size, shuffle=False, pin_memory=True,num_workers=4)
    
    opt.data_iter = iter(opt.train_loader)
    opt.real_iter = iter(opt.real_loader)
    opt.shot_iter = iter(opt.shot_loader)

    return opt

def main(opt):
    
    # init
    opt = init_settings(opt)
    for epoch in range(opt.epochs):
        
        train(opt, epoch)
        if epoch % 100 == 0:
            validate(opt, epoch)

    opt.writer.close()


if __name__ == "__main__":
    opt = parse_opts()
    opt.current_folder = os.getcwd()
    try:
        main(opt)
    except KeyboardInterrupt:
        print("ctrl + c ")
