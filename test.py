from dataloader.dataset import *
from utils.utility import *
from utils.resume import *
from models.model import *
from loss.seg_loss import *
from torch.autograd import Variable
from scipy import ndimage
import copy
import cv2

from argparse import *
import yaml

class StreamSegMetrics():
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwav IOU
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW IoU": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

def write_seq_images(opt, input_imgs, input_mask, predicted, idx,key,metrics):
    def _func(x):
        return (x.numpy()*255).astype(np.uint8).transpose(1,2,0).squeeze()

    base_num = idx * input_imgs.shape[0]

    mask_refine = opt.writer.refiner.bin(predicted)
    
    input_imgs = scale_img_back(input_imgs, device=opt.gpu)
    input_imgs, input_mask, predicted = input_imgs.cpu(), input_mask.cpu(), predicted.cpu()
    mask_refine = mask_refine.cpu()

    mask_raw = copy.deepcopy(predicted)
    mask_raw[mask_raw >= 0.5] = 1
    mask_raw[mask_raw < 0.5] = 0

    mask_out = input_imgs * mask_refine 

    for i, (img, mask, out, maskout,raw,refine) in enumerate(zip(input_imgs, input_mask, predicted,mask_out,mask_raw,mask_refine)):
        
        metrics.update(mask.numpy().transpose(1,2,0).squeeze(),refine.numpy().transpose(1,2,0).squeeze())

        img = cv2.cvtColor(_func(img), cv2.COLOR_RGB2BGR)
        maskout = cv2.cvtColor(_func(maskout), cv2.COLOR_RGB2BGR)
        mask = _func(mask)
        out = _func(out)
        raw = _func(raw)
        refine = _func(refine)


        dump_folder = os.path.join(opt.dump_folder,str(base_num+i))

        if not os.path.exists(dump_folder):
            os.mkdir(dump_folder)

        cv2.imwrite(join(dump_folder, key+"_img_%04d"%(base_num+i)+ ".png"), img)
        cv2.imwrite(join(dump_folder, key+"_gt_%04d"%(base_num+i)+ ".png"), mask)
        cv2.imwrite(join(dump_folder, key+"_refine_%04d"%(base_num+i)+ ".png"), refine)
        cv2.imwrite(join(dump_folder, key+"_maskout_%04d"%(base_num+i)+ ".png"), maskout)

            
def validate(opt,key):
    metrics = StreamSegMetrics(2)
    metrics.reset()
    opt.G.eval()
    opt.writer.reset()
    with torch.no_grad():
        for idx, (input_imgs, input_mask) in enumerate(opt.val_loader):
            
            input_imgs = input_imgs.cuda(opt.gpu)
            predicted = opt.G(input_imgs)
            write_seq_images(opt, input_imgs, input_mask, predicted, idx,key,metrics)

            predicted_ = predicted.cpu().numpy().repeat(3,axis=1)
            input_imgs = input_imgs.cpu().numpy()
            
        score = metrics.get_results()
        print(metrics.to_str(score))

def folder_init(opt):
    ''' tensorboard initialize: create tensorboard filename based on time tick and hyper-parameters

        Args:
            opt: parsed options from cmd or .yml(in config/ folder)
            
        Returns:
            opt: add opt.dump_folder and return opt
        
    '''    
    
    return opt

def init_settings(opt, data_path):
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(1124)
    opt.gpu = torch.device(opt.device_id)
    opt.dice_criterion = DiceLoss().cuda(opt.gpu)
    opt.bce_criterion = nn.BCEWithLogitsLoss().cuda(opt.gpu)
    opt.self_op = SelfOperation(opt)

    opt = init_models(opt)
    opt = init_dataset(opt, data_path)
    opt = folder_init(opt)
    opt.writer = TensorWriter(opt)
    
    opt.ones = torch.ones(opt.batch_size).cuda(opt.gpu)
    opt.zeros = torch.zeros(opt.batch_size).cuda(opt.gpu)

    
    return opt

def init_models(opt):
    # model init
    generator = ModelFactory(opt)
    generator.register_hook_model(get_generator_model)
    generator.register_hook_optimizer(get_ralamb_optimizer)

    opt.G, opt.optim_G, opt.sche_G = generator()

    return opt


def init_dataset(opt, data_path):

    opt.portrait_val = SegmentationDataset(opt, split="test", folder=data_path)
    opt.val_loader =  DataLoader(opt.portrait_val, collate_fn=segmentation_collate_fn, batch_size=opt.batch_size, shuffle=False, pin_memory=True,num_workers=4)
    return opt

## option args
def parse_opts():
    parser = ArgumentParser(description="segmentation")
    parser.add_argument('--hyper_setting', default='config/dynamic_config.yml',type=str,help="hyper-parameters of experiments")
    parser.add_argument('--device_id', default=2, type=int,help="CUDA device.")
    parser.add_argument('--comments', default="mask_together", type=str)
    
    opt =  parser.parse_args()

    hyper_opt = yaml.safe_load(open(opt.hyper_setting,"r"))
    opt = Namespace(**hyper_opt,**vars(opt))
    return opt

def main():
    current_folder = os.getcwd()
    pretrain_model_dir = os.path.join(current_folder,'pretrain')
    dump_dir = os.path.join(current_folder,'result')
    key_list = ["pa"]  

    opt = parse_opts()
    
    data_folder = os.path.join(current_folder,'datasets')
    
    experiment_mapp = {
        'test': {'name':'test',
                 'data':data_folder},
    }
    opt = parse_opts()

    for k, item in experiment_mapp.items():

        experiment_name = item['name']
        data_path = item['data']

        experiment_name = experiment_name
        opt.experiment_name = experiment_name
        
        print("method-->",k)

        for key in key_list:  
            
            opt.dump_folder = join(dump_dir,k, key)

            if not os.path.exists(opt.dump_folder):
                os.makedirs(opt.dump_folder)

            opt = init_settings(opt,data_path)

            opt.G = Resume(pretrain_model_dir,experiment_name).resume_model(opt.G, model_path=None, key=key, state=False)
            opt.G.cuda(opt.gpu)
            validate(opt,key)


    opt.writer.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("ctrl + c ")
