from utils import *
import matplotlib.pyplot as plt
import os
import shutil
import os.path as osp
import torch.utils.data as Data
from PIL import Image, ImageOps, ImageFilter
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class IRSTD_Dataset(Data.Dataset):
    def __init__(self, args, mode='train'):
        
       dataset_dir = args.dataset_dir

       if mode == 'train':
          txtfile = 'img_idx/train.txt'
          #txtfile = 'img_idx/train_NUDT-SIRST.txt'
       elif mode == 'val':
          #txtfile = 'img_idx/test_NUDT-SIRST.txt'
          txtfile = 'img_idx/test.txt'

       self.list_dir = osp.join(dataset_dir, txtfile)
       self.imgs_dir = osp.join(dataset_dir, 'images')
       self.label_dir = osp.join(dataset_dir, 'masks')

       self.names = []
       with open(self.list_dir, 'r') as f:
           self.names += [line.strip() for line in f.readlines()]
       self.mode = mode
       self.crop_size = args.crop_size
       self.base_size = args.base_size
       self.transform = transforms.Compose([
           transforms.ToTensor(),
           #transforms.Normalize([.485, .456, .406], [.229, .224, .225]), #三通道
           transforms.Normalize([-0.1246], [1.0923]), # mean and std 单通道
       ])
    
    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')     
        
        label_path = osp.join(self.label_dir, name+'.png')
        #img = Image.open(img_path).convert('RGB')
        img = Image.open(img_path).convert('L')
        mask = Image.open(label_path).convert('1')
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        
        img, mask = self.transform(img), transforms.ToTensor()(mask)
        #return img, mask,name
        return img, mask       #原本为这一行
    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):   #训练同步变换方法
        # random mirror
        if random.random() < 0.5:   # 有50%的概率执行以下操作
            img = img.transpose(Image.FLIP_LEFT_RIGHT)  # 将图像进行左右翻转
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT) # 将掩码也进行左右翻转
        crop_size = self.crop_size   # 获取预设的裁剪尺寸  
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))  # 随机生成一个长边尺寸，范围在base_size的一半到两倍之间  
        w, h = img.size    # 获取图像的宽和高 
        if h > w:           # 根据图像的长宽比调整图像的尺寸，确保长边为long_size，同时计算短边的尺寸
            oh = long_size    # 设置新的高度为long_size 
            ow = int(1.0 * w * long_size / h + 0.5)   # 根据原图的宽高比计算新的宽度  
            short_size = ow  # 短边尺寸为新计算的高度
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)   # 使用双线性插值方法调整图像尺寸
        mask = mask.resize((ow, oh), Image.NEAREST)  # 使用最近邻插值方法调整掩码尺寸  
        # pad crop
        if short_size < crop_size:   # 如果调整后的短边尺寸小于预设的裁剪尺寸
            padh = crop_size - oh if oh < crop_size else 0  # 计算高度需要填充的像素数  
            padw = crop_size - ow if ow < crop_size else 0   # 计算宽度需要填充的像素数
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0) # 对图像和掩码进行填充，填充颜色为黑色（0值）
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size  # 再次获取调整并可能填充后的图像尺寸 
        x1 = random.randint(0, w - crop_size)   # 随机选择一个裁剪的起始点x坐标
        y1 = random.randint(0, h - crop_size)    # 随机选择一个裁剪的起始点y坐标  
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))   # 根据选择的起始点和预设的裁剪尺寸对图像和掩码进行裁剪 
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:   # 有50%的概率执行以下操作
            img = img.filter(ImageFilter.GaussianBlur(   # 对图像进行高斯模糊处理，模糊半径为随机数 
                radius=random.random()))
        return img, mask
    
class IRSTD_Dataset1(Data.Dataset):
    def __init__(self, args, mode='train'):
        dataset_dir = args.dataset_dir + '/' + args.dataset_name
        self.dataset_dir = dataset_dir
        if mode == 'train':
            txtfile = 'trainval.txt'
            #txtfile = 'img_idx/train_NUDT-SIRST.txt'
        elif mode == 'val':
            #txtfile = 'img_idx/test_NUDT-SIRST.txt'
            txtfile = 'test.txt'

        self.list_dir = osp.join(dataset_dir, txtfile)
        self.imgs_dir = osp.join(dataset_dir, 'images')
        self.label_dir = osp.join(dataset_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]
        self.mode = mode
        self.crop_size = args.crop_size
        self.base_size = args.base_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize([.485, .456, .406], [.229, .224, .225]), #三通道
            transforms.Normalize([-0.1246], [1.0923]), # mean and std 单通道
        ])
        
    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')             
        label_path = osp.join(self.label_dir, name+'.png')

        # txt文件的路径
        txt_path = (self.dataset_dir + '/descriptions/' + name +"_description" + '.txt').replace('//', '/')
        with open(txt_path, 'r', encoding='utf-8') as file:
            # 读取全部内容为字符串
            content = file.read()

        #img = Image.open(img_path).convert('RGB')
        img = Image.open(img_path).convert('L')
        mask = Image.open(label_path).convert('1')
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        
        img, mask = self.transform(img), transforms.ToTensor()(mask)
        #return img, mask,name
        return img, mask,content       #原本为这一行
    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):   #训练同步变换方法
        # random mirror
        if random.random() < 0.5:   # 有50%的概率执行以下操作
            img = img.transpose(Image.FLIP_LEFT_RIGHT)  # 将图像进行左右翻转
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT) # 将掩码也进行左右翻转
        crop_size = self.crop_size   # 获取预设的裁剪尺寸  
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))  # 随机生成一个长边尺寸，范围在base_size的一半到两倍之间  
        w, h = img.size    # 获取图像的宽和高 
        if h > w:           # 根据图像的长宽比调整图像的尺寸，确保长边为long_size，同时计算短边的尺寸
            oh = long_size    # 设置新的高度为long_size 
            ow = int(1.0 * w * long_size / h + 0.5)   # 根据原图的宽高比计算新的宽度  
            short_size = ow  # 短边尺寸为新计算的高度
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)   # 使用双线性插值方法调整图像尺寸
        mask = mask.resize((ow, oh), Image.NEAREST)  # 使用最近邻插值方法调整掩码尺寸  
        # pad crop
        if short_size < crop_size:   # 如果调整后的短边尺寸小于预设的裁剪尺寸
            padh = crop_size - oh if oh < crop_size else 0  # 计算高度需要填充的像素数  
            padw = crop_size - ow if ow < crop_size else 0   # 计算宽度需要填充的像素数
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0) # 对图像和掩码进行填充，填充颜色为黑色（0值）
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size  # 再次获取调整并可能填充后的图像尺寸 
        x1 = random.randint(0, w - crop_size)   # 随机选择一个裁剪的起始点x坐标
        y1 = random.randint(0, h - crop_size)    # 随机选择一个裁剪的起始点y坐标  
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))   # 根据选择的起始点和预设的裁剪尺寸对图像和掩码进行裁剪 
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:   # 有50%的概率执行以下操作
            img = img.filter(ImageFilter.GaussianBlur(   # 对图像进行高斯模糊处理，模糊半径为随机数 
                radius=random.random()))
        return img, mask


    def _testval_sync_transform(self, img, mask):
        # 测试时直接将图片resize为basesize大小
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask




class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        super(TrainSetLoader).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        if not os.path.exists(self.dataset_dir +'/img_idx/train_' + dataset_name + '.txt') and os.path.exists(self.dataset_dir +'/img_idx/train.txt'):
            shutil.copyfile(self.dataset_dir +'/img_idx/train.txt', self.dataset_dir +'/img_idx/train_' + dataset_name + '.txt')
        with open(self.dataset_dir +'/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()
        
    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//','/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.png').replace('//','/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.bmp').replace('//','/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.bmp').replace('//','/'))
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
            
        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5) 
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        return img_patch, mask_patch
    def __len__(self):
        return len(self.train_list)

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        
    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//','/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.png').replace('//','/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//','/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.bmp').replace('//','/'))

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
        
        h, w = img.shape
        img = PadImg(img)
        mask = PadImg(mask)
        
        img, mask = img[np.newaxis,:], mask[np.newaxis,:]
        
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask, [h,w], self.test_list[idx]
    def __len__(self):
        return len(self.test_list) 
    
class TrainSetLoader1(Dataset):
    # 训练图片大小是256，可以自己定义
    # 好处是训练图片没有resize操作，避免了信息丢失
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        super(TrainSetLoader).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()

        

    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//', '/')).convert(
                'I')  # read image base on version ”I“
            # img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//','/'))
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.bmp').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.bmp').replace('//', '/'))
        
        # txt文件的路径
        txt_path = (self.dataset_dir + '/descriptions/' + self.train_list[idx] +"_description" + '.txt').replace('//', '/')
        with open(txt_path, 'r', encoding='utf-8') as file:
            # 读取全部内容为字符串
            content = file.read()
        # 图片归一化
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)  # convert PIL to numpy  and  normalize
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)  # 把短的一边先pad至256 把长的一边 随机裁出256  输出 256 256

        img_patch, mask_patch = self.tranform(img_patch, mask_patch)  # 数据翻转增强
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]  # 升维
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))  # numpy 转tensor
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))  # numpy 转tensor
        return img_patch, mask_patch,content

    def __len__(self):
        return len(self.train_list)
    
class TrainSetLoader2(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size

        if dataset_name == 'IRSTD-1K':#修改
            self.patch_size = 512
            
        with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()

        

    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//', '/')).convert(
                'I')  # read image base on version ”I“
            # img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//','/'))
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.bmp').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.bmp').replace('//', '/'))
        
        # txt文件的路径
        txt_path = (self.dataset_dir + '/descriptions/' + self.train_list[idx] +"_description" + '.txt').replace('//', '/')
        with open(txt_path, 'r', encoding='utf-8') as file:
            # 读取全部内容为字符串
            content = file.read()
        # 图片归一化
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)  # convert PIL to numpy  and  normalize
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)  # 把短的一边先pad至256 把长的一边 随机裁出256  输出 256 256

        img_patch, mask_patch = self.tranform(img_patch, mask_patch)  # 数据翻转增强
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]  # 升维
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))  # numpy 转tensor
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))  # numpy 转tensor
        return img_patch, mask_patch,content

    def __len__(self):
        return len(self.train_list)
    
class TrainSetLoader3(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size

        if dataset_name == 'IRSTD-1K':#修改
            self.patch_size = 512
            
        with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()

        

    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//', '/')).convert(
                'I')  # read image base on version ”I“
            # img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//','/'))
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.bmp').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.bmp').replace('//', '/'))
        
        # txt文件的路径
        txt_path = (self.dataset_dir + '/descriptions0/' + self.train_list[idx] +"_description" + '.txt').replace('//', '/')
        with open(txt_path, 'r', encoding='utf-8') as file:
            # 读取全部内容为字符串
            content = file.read()
        # 图片归一化
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)  # convert PIL to numpy  and  normalize
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)  # 把短的一边先pad至256 把长的一边 随机裁出256  输出 256 256

        img_patch, mask_patch = self.tranform(img_patch, mask_patch)  # 数据翻转增强
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]  # 升维
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))  # numpy 转tensor
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))  # numpy 转tensor
        return img_patch, mask_patch,content

    def __len__(self):
        return len(self.train_list)
    
class TrainSetLoader4(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size

        if dataset_name == 'IRSTD-1K':#修改
            self.patch_size = 512
            
        with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()

        

    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//', '/')).convert(
                'I')  # read image base on version ”I“
            # img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//','/'))
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.bmp').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.bmp').replace('//', '/'))
        
        # txt文件的路径
        txt_path = (self.dataset_dir + '/descriptions1/' + self.train_list[idx] +"_description" + '.txt').replace('//', '/')
        with open(txt_path, 'r', encoding='utf-8') as file:
            # 读取全部内容为字符串
            content = file.read()
        # 图片归一化
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)  # convert PIL to numpy  and  normalize
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)  # 把短的一边先pad至256 把长的一边 随机裁出256  输出 256 256

        img_patch, mask_patch = self.tranform(img_patch, mask_patch)  # 数据翻转增强
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]  # 升维
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))  # numpy 转tensor
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))  # numpy 转tensor
        return img_patch, mask_patch,content

    def __len__(self):
        return len(self.train_list)
    
class TestSetLoader1(Dataset):
    # 测试图片的大小是原图最近的32的倍数
    # 测试图片同样没有resize操作
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.bmp').replace('//', '/'))

        # txt文件的路径
        txt_path = (self.dataset_dir + '/descriptions/' + self.test_list[idx] +"_description" + '.txt').replace('//', '/')
        with open(txt_path, 'r', encoding='utf-8') as file:
            # 读取全部内容为字符串
            content = file.read()

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        # if mask.shape == (416,608):
        #     print('111')
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        h, w = img.shape
        # 将图片pad到32的倍数
        img = PadImg(img)
        mask = PadImg(mask)

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        if img.size() != mask.size():
            print('111')
        # return img, mask, [h, w], self.test_list[idx]
        return img, mask,content

    def __len__(self):
        return len(self.test_list)
    
class TestSetLoader2(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name,patch_size, img_norm_cfg=None):
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        self.patch_size = patch_size
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.bmp').replace('//', '/'))

        # txt文件的路径
        txt_path = (self.dataset_dir + '/descriptions/' + self.test_list[idx] +"_description" + '.txt').replace('//', '/')
        with open(txt_path, 'r', encoding='utf-8') as file:
            # 读取全部内容为字符串
            content = file.read()


        base_size = self.patch_size
        # print(base_size)
        # # 如果 base_size 是元组，提取第一个元素
        # if isinstance(base_size, tuple):
        #     base_size = base_size[0]
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)
        
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        # if mask.shape == (416,608):
        #     print('111')
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        if img.size() != mask.size():
            print('111')
        # return img, mask, [h, w], self.test_list[idx]
        return img, mask,content

    def __len__(self):
        return len(self.test_list)

class InferenceSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(InferenceSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        
    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//','/')).convert('I')
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//','/')).convert('I')
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        
        h, w = img.shape
        img = PadImg(img)
        
        img = img[np.newaxis,:]
        
        img = torch.from_numpy(np.ascontiguousarray(img))
        return img, [h,w], self.test_list[idx]
    def __len__(self):
        return len(self.test_list) 


class EvalSetLoader(Dataset):
    def __init__(self, dataset_dir, mask_pred_dir, test_dataset_name, model_name):
        super(EvalSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.mask_pred_dir = mask_pred_dir
        self.test_dataset_name = test_dataset_name
        self.model_name = model_name
        with open(self.dataset_dir + "/" + self.test_dataset_name + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()

    def __getitem__(self, idx):
        mask_pred = Image.open((self.mask_pred_dir + self.test_dataset_name + '/' + self.model_name + '/' + self.test_list[idx] + '.png').replace('//','/'))
        mask_gt = Image.open(self.dataset_dir + "/" + self.test_dataset_name + '/masks/' + self.test_list[idx] + '.png')

        mask_pred = np.array(mask_pred, dtype=np.float32)  / 255.0
        mask_gt = np.array(mask_gt, dtype=np.float32)  / 255.0
        
        if len(mask_pred.shape) == 3:
            mask_pred = mask_pred[:,:,0]
        
        h, w = mask_pred.shape
        
        mask_pred, mask_gt = mask_pred[np.newaxis,:], mask_gt[np.newaxis,:]
        
        mask_pred = torch.from_numpy(np.ascontiguousarray(mask_pred))
        mask_gt = torch.from_numpy(np.ascontiguousarray(mask_gt))
        return mask_pred, mask_gt, [h,w]
    def __len__(self):
        return len(self.test_list) 


class augumentation(object):
    # 对输入图像 (input) 和对应的 mask (target) 做 随机翻转与随机转置
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random()<0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random()<0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input, target
