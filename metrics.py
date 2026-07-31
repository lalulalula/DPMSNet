import numpy as np
import torch
from skimage import measure
from scipy.spatial.distance import cdist

class ROCMetric():
    def __init__(self, nclass, bins):
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.target_arr = np.zeros(self.bins+1)
        self.t_arr = np.zeros(self.bins+1)

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            i_tp, i_pos, i_fp, i_neg, i_target, i_t = cal_tp_pos_fp_neg_target(preds, labels, self.nclass, score_thresh)

            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg
            self.target_arr[iBin] += i_target
            self.t_arr[iBin] += i_t

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)
        tar = self.target_arr / (self.t_arr)

        return tp_rates, fp_rates, tar
    
def cal_tp_pos_fp_neg_target(output, target, nclass, score_thresh):
    mini = 1
    maxi = 1 # nclass
    nbins = 1 # nclass

    predict = np.array((output > score_thresh).cpu()).astype('int64') # P
    target = target.detach().numpy().astype('int64').squeeze()  # T
    intersection = predict * (predict == target) # TP
    tp = intersection.sum()
    fp = (predict * (predict != target)).sum()  # FP
    tn = ((1 - predict) * (predict == target)).sum()  # TN
    fn = ((predict != target) * (1 - predict)).sum()   # FN
    pos = tp + fn
    neg = fp + tn
    
    labels = measure.label(target,connectivity=2) #
    # stats
    properties = measure.regionprops(labels)

    t = labels.max()
    tar = 0
    m,n = target.shape
    for prop in properties:
        center = prop.centroid

        r = round(center[1])
        c = round(center[0])

        distanceMap = np.zeros((m, n));
        distanceMap[ c - 2 : c,r - 2 : r ] = 1;
    
        if sum(sum(predict * predict * distanceMap)).any() > 0:
            tar += 1
    
    return tp, pos, fp, neg, tar, t
       
class mIoU():
    
    def __init__(self):
        super(mIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')
        # 被正确预测的标签像素和标签像素总数
        correct, labeled = batch_pix_accuracy(preds, labels)
        # 交集和并集像素数
        inter, union = batch_intersection_union(preds, labels)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return float(pixAcc), mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

class PD_FA():
    def __init__(self,):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target= 0
        self.pre_target= 0
        self.dismatch_target= 0
    def update(self, preds, labels, size):
        predits  = np.array((preds).cpu()).astype('int64')
        labelss = np.array((labels).cpu()).astype('int64') 

        # 连通域分析
        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss , connectivity=2)
        coord_label = measure.regionprops(label)

        pre_target = len(coord_image)

        # 获取真实目标数
        self.target    += len(coord_label)

        # 获取预测目标数
        self.pre_target    += pre_target

        self.image_area_total = []
        self.distance_match   = []
        self.dismatch         = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        true_img = np.zeros(predits.shape)
        # 对真实目标连通域进行遍历
        for i in range(len(coord_label)):
            # 获得真实目标质心
            centroid_label = np.array(list(coord_label[i].centroid))
            # 遍历预测连通域
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                # 获取预测目标与真实目标质心的距离
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                # 如果距离小于 3，认为这两个目标是匹配的
                if distance < 3:
                    self.distance_match.append(distance)
                    true_img[coord_image[m].coords[:,0], coord_image[m].coords[:,1]] = 1
                    # 删除当前已匹配的预测目标区域
                    del coord_image[m]
                    break
        # 虚警的目标数，用于计算目标级虚警
        self.dismatch_target += len(coord_image)        
        # self.dismatch_target += ( pre_target - len(self.distance_match) )

        # 不匹配的像素数
        self.dismatch_pixel += (predits - true_img).sum()
        # 总像素数
        self.all_pixel +=size[0]*size[1]
        # 匹配成功的目标数，用于计算PD
        self.PD +=len(self.distance_match)

        

        # print("hello")



    def get(self):
        # 预测出的像素中，不是真实目标像素的比例，即虚警率（像素级）
        Final_FA =  self.dismatch_pixel / self.all_pixel
        # 真实目标中，被成功预测的比例，及检测率（目标级）
        Final_PD =  self.PD /self.target
        # return Final_PD, float(Final_FA.cpu().detach().numpy())
        return Final_PD, float(Final_FA)
    
    def get_FAT(self):
        # 预测出的目标中，不是真实目标的比例，即虚警率（目标级）
        Final_FAT =  self.dismatch_target / self.target
        return Final_FAT

    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])

def batch_pix_accuracy(output, target):   
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    # 将预测结果转换为二进制值（值大于 0 的为 1，否则为 0）
    predict = (output > 0).float()
    # 计算标签中大于 0 的像素总数（即有标签的像素数量）
    pixel_labeled = (target > 0).float().sum()
    # 计算预测与真实标签一致且真实标签大于 0 的像素数（即正确预测的有标签像素）
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"   
    return pixel_correct, pixel_labeled

def batch_intersection_union(output, target):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    # 分别统计交集、预测值、标签中为1的像素数量
    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    # 并集像素为1的数量
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union


class PD_FA_new():
    def __init__(self):
        super(PD_FA_new, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.FAT = 0
        self.target = 0

    def update(self, preds, labels, size):
        predits  = np.array((preds).cpu()).astype('int64')
        labelss = np.array((labels).cpu()).astype('int64')

        # -------------------- 区域检测 --------------------
        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss , connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)
        self.image_area_total    = []
        self.image_area_match    = []
        self.image_area_mismatch = []
        self.distance_match      = []
        self.distance_mismatch   = []
        self.dismatch            = []

        for K in range(len(coord_image)):
            self.image_area_total.append(coord_image[K].area)

        # -------------------- PD 逻辑 --------------------
        matched_flags = np.zeros(len(coord_image), dtype=bool)
        true_img = np.zeros(predits.shape)

        for i, gt in enumerate(coord_label):
            centroid_label = np.array(list(gt.centroid))
            for m, pr in enumerate(coord_image):
                if matched_flags[m]:
                    continue
                centroid_image = np.array(list(pr.centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                if distance <= 3:
                    self.distance_match.append(distance)
                    true_img[pr.coords[:, 0], pr.coords[:, 1]] = 1
                    matched_flags[m] = True
                    break

        # -------------------- FAT 逻辑（聚合 cluster） --------------------
        pred_boundaries = [r.coords for r in coord_image]  # 所有预测目标的像素坐标
        fat_flags = np.zeros(len(coord_image), dtype=bool)

        for i, coords_i in enumerate(pred_boundaries):
            if fat_flags[i]:
                continue
            cluster = [i]
            fat_flags[i] = True
            for j, coords_j in enumerate(pred_boundaries):
                if fat_flags[j]:
                    continue
                # 边界距离 ≤3 的预测目标聚合为同一个 cluster
                dist_matrix = cdist(coords_i, coords_j)
                if np.min(dist_matrix) <= 3:
                    cluster.append(j)
                    fat_flags[j] = True

            # 判断 cluster 是否匹配 GT
            is_matched = False
            for idx in cluster:
                pr_coords = pred_boundaries[idx]
                for gt in coord_label:
                    gt_coords = gt.coords
                    if np.min(cdist(pr_coords, gt_coords)) <= 3:
                        is_matched = True
                        break
                if is_matched:
                    break

            if not is_matched:
                self.FAT += 1  # 这个 cluster 计作 1 个 FAT

        # -------------------- 像素级虚警 --------------------
        self.dismatch_pixel += (predits - true_img).sum()
        self.all_pixel += size[0] * size[1]
        self.PD += len(self.distance_match)

    def get(self):
        Final_FA = self.dismatch_pixel / self.all_pixel
        if self.target == 0:
            Final_PD = 0
            Final_FAT = 0
        else:
            Final_PD = self.PD / self.target
            Final_FAT = self.FAT / self.target
        return Final_PD, float(Final_FA), Final_FAT

    def reset(self):
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.FAT = 0
        self.target = 0

