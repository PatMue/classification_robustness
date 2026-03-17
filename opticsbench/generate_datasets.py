"""
If you find this useful in your research, please cite:

@article{mueller2023_opticsbench,
	author   = {Patrick Müller, Alexander Braun and Margret Keuper},
	title    = {Classification robustness to common optical aberrations},
	journal  = {Proceedings of the International Conference on Computer Vision Workshops (ICCVW)},
	year     = {2023}
}

"""
__license__ = "MIT-license"
__author__ = "Patrick M\"uller"

import os
import sys
import random
import argparse
from tqdm import tqdm
import logging

import torch
from torch.nn.parallel import parallel_apply
from torchvision import datasets,transforms
from PIL import Image

sys.path.append(os.path.abspath(os.path.join("..","..","optics_augment")))
from optics_augment.augment import OpticsAugment
from utils import ImageNetPathsDataset

logger = logging.getLogger(__name__)


class CreateBenchmark():
    """
    This generates for a given "clean data folder" --> ImageNette/val
        above folder structure
    should be like:
        images
            ImageNet
                /val
    creates:
        images
            ImageNet
                /val
                /corruptions
                    ...

    """
    __seed__ = 0
    __params__ = [3,4,5,6,7,8,9,10]
    __modes__ = {"defocus_spherical":[3,8],\
        "astigmatism":[4,5],
        "coma":[6,7],
        "trefoil":[9,10]
        }

    quality = 95


    def __init__(self,testdata_path,psfstack_path=None,all_modes=True,\
            batch_size=50,severities=None,padding_mode="zeros"):
        """
        testdata_path: e.g. ..,ImageNet/val
        """
        print("put all images in a folder data/<dataset_name>/val")
        self.rootdir,__ = os.path.split(testdata_path)
        self.isgpu = torch.cuda.is_available()
        self.device = 'cuda' if self.isgpu else 'cpu'
        self.psf_stack = self._load_blur_kernel_stack(psfstack_path,device=self.device)
        cname = "corruptions"
        corruptions_name = cname+"_rg" if psfstack_path.__contains__("kernel_stack_rg_iccvw.pt") else cname
        self.paths = {"val":testdata_path,\
            "corruptions":os.path.join(self.rootdir,corruptions_name)}

        print(f"Loaded: {psfstack_path}")

        self.create_benchmark_paths()
        self.batch_size = batch_size
        self.severities = severities if severities is not None else [1,2,3,4,5]
        self.padding_mode = padding_mode

        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()#,
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        self.save_transform = transforms.ToPILImage()

        dataset = datasets.ImageFolder(self.paths["val"])
        self.dataset = ImageNetPathsDataset(dataset,self.val_transform)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,\
            shuffle=False,
            batch_size = self.batch_size,
            pin_memory=self.isgpu)


    @staticmethod
    def _load_blur_kernel_stack(psfstack_path=None,device='cpu'):
        """ 
        psfstack_path: path to kernel stack, if None, will be searched in parent folder (default)
        """
        if psfstack_path is None and __name__ == "__main__":
            psfstack_path = os.path.abspath(os.path.join(os.getcwd(),"..","kernel_stack_iccvw.pt"))
        kernels = torch.load(psfstack_path)
        is_gpu = "cuda" if torch.cuda.is_available() else "cpu"
        kernels = kernels.to(device)
        if kernels.ndim == 5:
            kernels = kernels.unsqueeze(2)  # downward compatibility
        print(f"loaded kernel stack of shape: {kernels.shape} ({is_gpu}),\n{psfstack_path}")
        return kernels


    def create_benchmark_paths(self):

        for p in self.__modes__:
            outdir = os.path.join(self.paths['corruptions'],p)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            self.paths[p] = outdir

        print(f"created folder hierachy in {self.paths['corruptions']}")


    def run(self):
        """"""
        for severity in tqdm(self.severities,total=len(self.severities),\
                desc="Severity: ",leave=True):
            for p in tqdm(self.__modes__,desc="Mode: ",leave=False):
                random.seed(self.__seed__,version=2) # every time the same
                params = self.__modes__[p]
                _params = [random.choice(params)-3 for n in range(self.batch_size)]
                outdir = os.path.join(self.paths[p],str(severity))
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                with torch.no_grad():
                    for i,(images,fpaths) in enumerate(self.dataloader):
                        images = images.to(self.device)
                        images = self.__gen__(images,_params,severity=severity).cpu()
                        self._create_folders(fpaths,outdir)
                        parallel_apply([self._save_image]*len(images),\
                            [(im,fpath,outdir) for im,fpath in zip(images,fpaths)])


    def __gen__(self,images,_params,severity=3):
        _batchsize = images.shape[0]
        _chroma = 0
        kernels = self.psf_stack[:, severity-1, _chroma,:]
        return torch.stack(\
            parallel_apply([OpticsAugment._apply]*len(images),\
            [(im,kernels[_param,:].unsqueeze(0),self.padding_mode) for im,_param in zip(images,_params)]),\
            dim=0).squeeze()


    def _create_folders(self,fpaths,outdir):
        for fpath in fpaths:
            fpath = os.path.normpath(fpath)
            folder,imname = os.path.split(fpath)
            __,folder = os.path.split(folder)
            outpath = os.path.join(outdir,folder)
            if not os.path.exists(outpath):
                os.makedirs(outpath)


    def _save_image(self,image,fpath,outdir):
        fpath = os.path.normpath(fpath)
        folder,imname = os.path.split(fpath)
        __,folder = os.path.split(folder)
        savepath = os.path.join(outdir,folder,imname)
        transforms.ToPILImage()(torch.clip(image,min=0.0,max=1.0)).save(savepath,quality=self.quality)



def create_benchmark(**kwargs):
    psfstack_path = os.path.abspath(os.path.join(os.getcwd(),"..","kernel_stack"))
    if kwargs.get("use_rg_stack",None):
        psfstack_path += "_rg_iccvw.pt"
    else:
        psfstack_path += "_iccvw.pt"		
        
    testdata_path = kwargs.get("testdata_path") # the path to image val
    batch_size = kwargs.get("batch_size",64)
    benchmark_creator = CreateBenchmark(
                testdata_path,
                psfstack_path=psfstack_path,\
                batch_size=batch_size,\
                severities=kwargs.get("severities",[1,2,3,4,5]),
                padding_mode=kwargs.get("padding_mode","reflect")
            )
    benchmark_creator.run()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("-t","--testdata_path",default="",type=str)
    argparser.add_argument("--padding_mode",default="zeros",type=str)
    argparser.add_argument("--use_rg_stack",action="store_true",default=False,\
        help="create the OpticsBenchRG ... magenta red and green only")
    argparser.add_argument("--severities",default=[1,2,3,4,5],type=int,nargs='+')

    kwargs = argparser.parse_args().__dict__

    assert os.path.exists(kwargs['testdata_path'])
    create_benchmark(**kwargs)