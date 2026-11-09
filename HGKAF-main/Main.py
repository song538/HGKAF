import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from matplotlib import cm
import spectral as spy
from sklearn import metrics
import time
from sklearn import preprocessing
import torch
import LDA_SLIC
import HCGN
from torchsummary import summary
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import spectral as spy
from sklearn import preprocessing
from sklearn.manifold import TSNE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# FLAG =1, indian
# FLAG =2, paviaU
# FLAG =3, salinas
# ========== 1. 选择数据集 ==========
samples_type = ['ratio', 'same_num'][0]
# for (FLAG, curr_train_ratio) in [(1,5),(1,10),(1,15),(1,20),(1,25),
# (2,5),(2,10),(2,15),(2,20),(2,25),
# (3,5),(3,10),(3,15),(3,20),(3,25)]:
# for (FLAG, curr_train_ratio,Scale,Scale1,Scale2) in [(1,0.03,10,50,200)]:                                 #0.002
for (FLAG, curr_train_ratio,Scale,Scale1,Scale2) in [(6,0.02,10,100,150)]:   # IP [1,0.02,10,100,150]  Pu [2,0.002,50,100,200] LK[5,0.001,50,100,150]
# for (FLAG, curr_train_ratio,Scale) in [(2,0.01,100),(3,0.01,100)]:
    torch.cuda.empty_cache()
    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    AVG_ALL = []
    Train_Time_ALL=[]
    Test_Time_ALL=[]
    
    # Seed_List=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]#随机种子点[73]
    # Seed_List=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19][85]
    # Seed_List=[0,1,2]
    # Seed_List=[60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89]
    Seed_List=[73]
    if FLAG == 1:
        data_mat = sio.loadmat('/root/IP/Indian_pines_corrected.mat')
        # print(data_mat)
        data = data_mat['data']
        gt_mat = sio.loadmat('/root/IP/Indian_pines_gt.mat')
        gt = gt_mat['groundT']
        # 参数预设
        # train_ratio = 0.05  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.03  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 16  # 样本类别数
        learning_rate = 5e-4   # 学习率
        max_epoch =1300  # 迭代次数
        dataset_name = "indian_"  # 数据集名称
        # superpixel_scale=100
        pass
    if FLAG == 2:
        data_mat = sio.loadmat('/userData/gpusjl/PU/PaviaU.mat')
        data = data_mat['data']
        gt_mat = sio.loadmat('/userData/gpusjl/PU/PaviaU_gt.mat')
        # print(gt_mat.keys())
        gt = gt_mat['groundT']
        
        # 参数预设
        # train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = curr_train_ratio  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类  0.5%
        class_count = 9  # 样本类别数
        learning_rate = 5e-4  # 学习率
        max_epoch = 200  # 迭代次数
        dataset_name = "PaviaU_"  # 数据集名称
        # superpixel_scale = 100
        pass
    if FLAG == 3:
        data_mat = sio.loadmat('datasets/Salinas_corrected.mat')
        data = data_mat['salinas_corrected']
        gt_mat = sio.loadmat('datasets/Salinas_gt.mat')
        gt = gt_mat['salinas_gt']
        
        # 参数预设
        # train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.005  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 16  # 样本类别数
        learning_rate = 1e-4  # 学习率
        max_epoch = 600  # 迭代次数
        dataset_name = "salinas_"  # 数据集名称
        # superpixel_scale = 100
        pass
    if FLAG == 4:
        data_mat = sio.loadmat('datasets/KSC.mat')
        data = data_mat['KSC']
        gt_mat = sio.loadmat('datasets/KSC_gt.mat')
        gt = gt_mat['KSC_gt']
        
        # 参数预设
        # train_ratio = 0.05  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 13  # 样本类别数
        learning_rate = 1e-3  # 学习率
        max_epoch = 600  # 迭代次数
        dataset_name = "KSC_"  # 数据集名称
        # superpixel_scale = 200
        pass

    if FLAG == 6:
        data_mat = sio.loadmat('/root/Houston /Houston.mat')
        # print(data_mat.keys())
        data = data_mat['Houston']
        gt_mat = sio.loadmat('/root/Houston /Houston_gt.mat')
        # print(gt_mat.keys())
        gt = gt_mat['Houston_gt']

        # 参数预设
        # train_ratio = 0.05  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.001  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 15  # 样本类别数
        learning_rate = 5e-4  # 学习率
        max_epoch = 600  # 迭代次数
        dataset_name = "HT"  # 数据集名称
        # superpixel_scale = 200
        pass


    if FLAG == 5:
        data_mat = sio.loadmat('/userData/gpusjl/LK/WHU_Hi_LongKou.mat')
        data = data_mat['data']
        gt_mat = sio.loadmat('/userData/gpusjl/LK/WHU_Hi_LongKou_gt.mat')
        gt = gt_mat['groundT']

        # 参数预设
        # train_ratio = 0.05  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.001  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 9  # 样本类别数
        learning_rate = 5e-4 # 学习率
        max_epoch = 200  # 迭代次数
        dataset_name = "WHU_Hi_LongKou"  # 数据集名称
        # superpixel_scale = 200
        pass

    if FLAG == 3:
        data_mat = sio.loadmat('C:/Users/Four Eleven/Desktop/code/datasets/WHU_Hi_HongHu.mat')
        data = data_mat['WHU_Hi_HongHu']
        gt_mat = sio.loadmat('C:/Users/Four Eleven/Desktop/code/datasets/WHU_Hi_HongHu_gt.mat')
        gt = gt_mat['WHU_Hi_HongHu_gt']

        # 参数预设
        # train_ratio = 0.05  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.001  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 22  # 样本类别数
        learning_rate = 5e-4  # 学习率
        max_epoch = 600  # 迭代次数
        dataset_name = "WHU_Hi_HongHu"  # 数据集名称
        # superpixel_scale = 200
        pass


    ###########
    superpixel_scale=Scale#########################
    superpixel_scale1=Scale1
    superpixel_scale2=Scale2
    train_samples_per_class = curr_train_ratio  # 当定义为每类样本个数时,则该参数更改为训练样本数
    val_samples = class_count
    train_ratio = curr_train_ratio
    cmap = cm.get_cmap('jet', class_count + 1)
    plt.set_cmap(cmap)
    m, n, d = data.shape  # 高光谱数据的三个维度

    # 数据standardization标准化,即提前全局BN
    orig_data=data
    height, width, bands = data.shape  # 原始高光谱数据的三个维度
    data = np.reshape(data, [height * width, bands])
    minMax = preprocessing.StandardScaler()
    data = minMax.fit_transform(data)
    data = np.reshape(data, [height, width, bands])
    
    # 打印每类样本个数
    gt_reshape=np.reshape(gt, [-1])
    for i in range(class_count):
        idx = np.where(gt_reshape == i + 1)[-1]
        samplesCount = len(idx)
        print(samplesCount)
    

    def GT_To_One_Hot(gt, class_count):
        '''
        Convet Gt to one-hot labels
        :param gt:
        :param class_count:
        :return:
        '''
        GT_One_Hot = []  # 转化为one-hot形式的标签
        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                temp = np.zeros(class_count,dtype=np.float32)
                if gt[i, j] != 0:
                    temp[int( gt[i, j]) - 1] = 1
                GT_One_Hot.append(temp)
        GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
        return GT_One_Hot

    def Draw_Classification_Map_Bg(label, name: str, scale: float = 4.0, dpi: int = 400, color_matrix=None, bg_color=[0,0,0]):
        """
        绘制分类图，只显示真实分类区域
        Args:
            label: 分类标签,其中-1表示未分类区域
            name: 保存路径和文件名
            scale: 图像缩放比例
            dpi: 输出图像的DPI
            color_matrix: 自定义颜色矩阵 [num_classes, 3]
            bg_color: 背景颜色RGB值，默认黑色[0,0,0]
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap
        
        # 创建掩码,只显示已分类区域
        mask = (label != -1)
        
        fig, ax = plt.subplots()
        numlabel = np.array(label)
        
        # 创建带掩码的标签数组
        masked_label = np.ma.masked_where(~mask, numlabel)
        
        if color_matrix is not None:
            # 使用自定义颜色矩阵
            color_matrix_norm = color_matrix / 255.0 if np.max(color_matrix) > 1 else color_matrix
            # 创建颜色映射
            cmap = ListedColormap(color_matrix_norm)
            # 设置未分类区域为指定背景色
            cmap.set_bad(np.array(bg_color)/255.0)
            plt.imshow(masked_label,
                    cmap=cmap,
                    interpolation='nearest')
        else:
            # 使用默认颜色映射
            plt.imshow(masked_label, 
                    interpolation='nearest')
            
        ax.set_axis_off()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        
        plt.savefig(name + '.pdf',
                    format='pdf', 
                    transparent=True,
                    dpi=dpi,
                    pad_inches=0,
                    bbox_inches='tight')
        plt.close()
    def visualize_test_tsne(features, labels, save_path, dataset_name):
            """
            测试集的 t-SNE 可视化
            features: [N, C] numpy array
            labels: [N] numpy array (label values 1..K)
            save_path: 保存图片完整路径（含文件名）
            dataset_name: 数据集名（用于标题）
            """
            try:
                import matplotlib.pyplot as plt
                from sklearn.manifold import TSNE
                # 定义颜色映射
                colormap = plt.cm.get_cmap('tab20', int(np.max(labels)) + 1)
                # 执行 t-SNE
                tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
                embeddings = tsne.fit_transform(features)
                # 可视化
                plt.figure(figsize=(12, 10))
                scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1],
                                    c=labels, cmap=colormap,
                                    alpha=0.8, s=10, edgecolors='none')
                cbar = plt.colorbar(scatter, ticks=np.unique(labels))
                cbar.set_label('Class ID')
                plt.title(f'Test Dataset t-SNE Visualization ({dataset_name})')
                plt.xlabel('t-SNE Dimension 1')
                plt.ylabel('t-SNE Dimension 2')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print("Warning: t-SNE visualization failed:", e)    




     # ===== 新增：在训练/评估之前定义打印数值混淆矩阵的函数（终端输出） =====
    def save_and_print_confusion_matrix(pred_map: np.ndarray,
                                        true_map: np.ndarray,
                                        class_count: int,
                                        save_prefix: str,
                                        normalize: bool = False,
                                        class_names: list = None):
        """
        计算并打印混淆矩阵（忽略 true_map 中为 0 或 -1 的背景像元），并保存图片/np文件到 save_prefix.*
        只在终端打印数值矩阵（用户要求）。
        """
        eps = 1e-12
        true_map_proc = true_map.copy()
        true_map_proc[true_map_proc < 1] = 0

        mask = (true_map_proc != 0)
        if mask.sum() == 0:
            print("No labeled pixels found in true_map. Confusion matrix not computed.")
            return None

        y_true = true_map_proc[mask].ravel()
        y_pred = pred_map[mask].ravel()

        labels = list(range(1, class_count + 1))
        cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)

        if normalize:
            denom = cm.sum(axis=1, keepdims=True).astype(float)
            cm = cm / (denom + eps)

        # 终端打印（数字矩阵）
        np.set_printoptions(precision=4, suppress=True)
        print("\nConfusion matrix (rows=true, cols=pred):")
        print(cm)
        # 同时保存为 .npy 供后续分析（可选）
        try:
            np.save(save_prefix + "_confusion.npy", cm)
        except Exception as e:
            print("Warning: failed to save confusion matrix:", e)

        # 也绘图保存一张图片（不影响终端输出）
        try:
            plt.figure(figsize=(max(6, class_count * 0.4), max(5, class_count * 0.3)))
            im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar(im, fraction=0.046, pad=0.04)
            tick_labels = class_names if class_names is not None else labels
            plt.xticks(np.arange(len(tick_labels)), tick_labels, rotation=45)
            plt.yticks(np.arange(len(tick_labels)), tick_labels)
            plt.ylabel('True')
            plt.xlabel('Predicted')
            plt.title('Confusion Matrix' + (' (normalized)' if normalize else ''))
            fmt = '.2f' if normalize else 'd'
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    val = cm[i, j]
                    plt.text(j, i, format(val, fmt),
                             ha="center", va="center",
                             color="white" if val > cm.max() / 2. else "black", fontsize=6)
            plt.tight_layout()
            plt.savefig(save_prefix + ( "_confusion_norm.png" if normalize else "_confusion.png"), dpi=200)
            plt.close()
        except Exception as e:
            print("Warning: failed to save confusion image:", e)

        return cm    
        
    for curr_seed in Seed_List:
        # step2:随机10%数据作为训练样本。方式：给出训练数据与测试数据的GT
        random.seed(curr_seed)
        np.random.seed(curr_seed)
        torch.manual_seed(curr_seed)
        torch.cuda.manual_seed_all(curr_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        gt_reshape = np.reshape(gt, [-1])
        print(gt_reshape.shape)
        gt1 = np.transpose(gt_reshape)
        sio.savemat(dataset_name+"gt.mat", {'gt': gt1})
        train_rand_idx = []
        val_rand_idx = []
        if samples_type == 'ratio':
            for i in range(class_count):
                idx = np.where(gt_reshape == i + 1)[-1]
                samplesCount = len(idx)
                rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
                rand_idx = random.sample(rand_list,
                                         np.ceil(samplesCount * train_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
                rand_real_idx_per_class = idx[rand_idx]
                train_rand_idx.append(rand_real_idx_per_class)
            # train_rand_idx = np.array(train_rand_idx)
            train_rand_idx = np.concatenate(train_rand_idx)
            train_data_index = list(train_rand_idx)  # 已是一维，直接用
            print("Number of training samples:", len(train_data_index))

            # 计算每个类别的训练样本个数
            class_train_counts = np.zeros(class_count)  # 创建一个数组，用于存储每类的训练样本数量

            for idx in train_data_index:
                class_label = gt_reshape[idx]  # 获取当前索引对应的类别标签
                if class_label != 0:  # 确保不是背景
                    class_train_counts[int(class_label) - 1] += 1  # 增加对应类别的计数

            # 打印每个类别的训练样本个数
            for i in range(class_count):
                print(f"Class {i + 1}: {class_train_counts[i]} training samples")
            # # print("train_rand_idx type:", type(train_rand_idx))
            # for c in range(train_rand_idx.shape[0]):
            #     a = train_rand_idx[c]
            #     for j in range(a.shape[0]):
            #         train_data_index.append(a[j])
            # train_data_index = np.array(train_data_index)
            # for c in range(train_rand_idx.shape[0]):
            #     a = train_rand_idx[c]
            #     # 保证 a 可迭代
            #     if isinstance(a, (np.ndarray, list, tuple)):
            #       for j in range(len(a)):
            #           train_data_index.append(a[j])
            #     else:
            #           train_data_index.append(a)
            # train_data_index = np.array(train_data_index)
            
            ##将测试集（所有样本，包括训练样本）也转化为特定形式
            train_data_index = set(train_data_index)
            all_data_index = [i for i in range(len(gt_reshape))]
            all_data_index = set(all_data_index)
            
            # 背景像元的标签
            background_idx = np.where(gt_reshape == 0)[-1]
            background_idx = set(background_idx)
            test_data_index = all_data_index - train_data_index - background_idx
            
            # 从测试集中随机选取部分样本作为验证集
            val_data_count = int(val_ratio * (len(test_data_index) + len(train_data_index)))  # 验证集数量
            val_data_index = random.sample(list(test_data_index), val_data_count)
            val_data_index = set(val_data_index)
            test_data_index = test_data_index - val_data_index  # 由于验证集为从测试集分裂出，所以测试集应减去验证集
            
            # 将训练集 验证集 测试集 整理
            test_data_index = list(test_data_index)
            train_data_index = list(train_data_index)
            val_data_index = list(val_data_index)
        
        if samples_type == 'same_num':
            for i in range(class_count):
                idx = np.where(gt_reshape == i + 1)[-1]
                samplesCount = len(idx)
                real_train_samples_per_class = train_samples_per_class
                rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
                if real_train_samples_per_class > samplesCount:
                    real_train_samples_per_class = samplesCount
                rand_idx = random.sample(rand_list,
                                         real_train_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
                rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
                train_rand_idx.append(rand_real_idx_per_class_train)
            train_rand_idx = np.array(train_rand_idx)
            val_rand_idx = np.array(val_rand_idx)
            train_data_index = []
            for c in range(train_rand_idx.shape[0]):
                a = train_rand_idx[c]
                for j in range(a.shape[0]):
                    train_data_index.append(a[j])
            train_data_index = np.array(train_data_index)
            
            train_data_index = set(train_data_index)
            all_data_index = [i for i in range(len(gt_reshape))]
            all_data_index = set(all_data_index)
            
            # 背景像元的标签
            background_idx = np.where(gt_reshape == 0)[-1]
            background_idx = set(background_idx)
            test_data_index = all_data_index - train_data_index - background_idx
            
            # 从测试集中随机选取部分样本作为验证集
            val_data_count = int(val_samples)  # 验证集数量
            val_data_index = random.sample(test_data_index, val_data_count)
            val_data_index = set(val_data_index)
            
            test_data_index = test_data_index - val_data_index
            # 将训练集 验证集 测试集 整理
            test_data_index = list(test_data_index)
            train_data_index = list(train_data_index)
            val_data_index = list(val_data_index)
        
        # 获取训练样本的标签图
        train_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(train_data_index)):
            train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
            pass
        
        # 获取测试样本的标签图
        test_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(test_data_index)):
            test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
            pass
        
        Test_GT = np.reshape(test_samples_gt, [m, n])  # 测试样本图
        
        # 获取验证集样本的标签图
        val_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(val_data_index)):
            val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
            pass

        train_samples_gt=np.reshape(train_samples_gt,[height,width])
        test_samples_gt=np.reshape(test_samples_gt,[height,width])
        val_samples_gt=np.reshape(val_samples_gt,[height,width])

        train_samples_gt_onehot=GT_To_One_Hot(train_samples_gt,class_count)
        test_samples_gt_onehot=GT_To_One_Hot(test_samples_gt,class_count)
        val_samples_gt_onehot=GT_To_One_Hot(val_samples_gt,class_count)

        train_samples_gt_onehot=np.reshape(train_samples_gt_onehot,[-1,class_count]).astype(int)
        test_samples_gt_onehot=np.reshape(test_samples_gt_onehot,[-1,class_count]).astype(int)
        val_samples_gt_onehot=np.reshape(val_samples_gt_onehot,[-1,class_count]).astype(int)

        ############制作训练数据和测试数据的gt掩膜.根据GT将带有标签的像元设置为全1向量##############
        # 训练集
        train_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        train_samples_gt = np.reshape(train_samples_gt, [m * n])
        for i in range(m * n):
            if train_samples_gt[i] != 0:
                train_label_mask[i] = temp_ones
        train_label_mask = np.reshape(train_label_mask, [m* n, class_count])

        # 测试集
        test_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        test_samples_gt = np.reshape(test_samples_gt, [m * n])
        for i in range(m * n):
            if test_samples_gt[i] != 0:
                test_label_mask[i] = temp_ones
        test_label_mask = np.reshape(test_label_mask, [m* n, class_count])

        # 验证集
        val_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        val_samples_gt = np.reshape(val_samples_gt, [m * n])
        for i in range(m * n):
            if val_samples_gt[i] != 0:
                val_label_mask[i] = temp_ones
        val_label_mask = np.reshape(val_label_mask, [m* n, class_count])
        
        
        ls = LDA_SLIC.LDA_SLIC(data, np.reshape( train_samples_gt,[height,width]), class_count-1)
        tic0=time.time()
        Q, S ,A,Seg= ls.simple_superpixel(scale=superpixel_scale)
        Q1, S1 ,A1,Seg1 = ls.simple_superpixel(scale=superpixel_scale1)
        Q2, S2 ,A2,Seg2 = ls.simple_superpixel(scale=superpixel_scale2)
        toc0 = time.time()
        LDA_SLIC_Time=toc0-tic0
        # np.save(dataset_name+'Seg',Seg)
        print("LDA-SLIC costs time: {}".format(LDA_SLIC_Time))
        Q=torch.from_numpy(Q).to(device)
        A=torch.from_numpy(A).to(device)
        Q1=torch.from_numpy(Q1).to(device)
        A1=torch.from_numpy(A1).to(device)
        Q2=torch.from_numpy(Q2).to(device)
        A2=torch.from_numpy(A2).to(device)

        #转到GPU
        train_samples_gt=torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
        test_samples_gt=torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
        val_samples_gt=torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)
        #转到GPU
        train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
        test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
        val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)
        #转到GPU
        train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
        test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
        val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)
        
        
        net_input=np.array( data,np.float32)
        net_input=torch.from_numpy(net_input.astype(np.float32)).to(device)

        if dataset_name == "indian_":
            net = HCGN.HCGN(height, width, bands, class_count, Q, A, Q1, A1, Q2, A2, model='smoothed')
        else:
            net = HCGN.HCGN(height, width, bands, class_count, Q, A, Q1, A1, Q2, A2)

        print("parameters", net.parameters(), len(list(net.parameters())))
        net.to(device)

        summary(net, (net_input.shape))
        # <-- 在此处插入：打印参数量与尝试计算 FLOPs（若安装 thop）
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Model params: {total_params} ({total_params/1e6:.3f}M), Trainable: {trainable_params} ({trainable_params/1e6:.3f}M)")

# ...existing code...
        try:
            # thop 计算 FLOPs（常用），若未安装会抛出异常
            from thop import profile
            # 在调用 profile 前，确保每个子模块都有 total_ops 属性，
            # 因为 thop 的 hook 会执行 m.total_ops += flops
            for module in net.modules():
                if not hasattr(module, 'total_ops'):
                    try:
                        # 优先以注册 buffer 的方式创建可就地加的 tensor
                        module.register_buffer('total_ops', torch.zeros(1))
                    except Exception:
                        # 回退：直接设置为数值（支持 += 运算）
                        setattr(module, 'total_ops', 0)

            # 构建一个符合模型输入格式的 dummy 输入（尝试两种常见排列）
            x = net_input.to(device).unsqueeze(0)  # [1, H, W, C] 或 [1, C, H, W]
            flops = None
            try:
                # 先尝试模型接受 (N, C, H, W)
                xin = x.permute(0, 3, 1, 2).contiguous()
                flops, params_thop = profile(net, inputs=(xin,), verbose=False)
            except Exception:
                # 回退：直接传入 (N, H, W, C)
                flops, params_thop = profile(net, inputs=(x,), verbose=False)
            print(f"FLOPs: {flops/1e9:.4f}G (estimated by thop)")
        except Exception as e:
            print("FLOPs not computed (install thop: pip install thop) or failed:", e)
# ...existing code...
        

        def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
            real_labels = reallabel_onehot
            we = -torch.mul(real_labels,torch.log(predict))
            we = torch.mul(we, reallabel_mask)
            pool_cross_entropy = torch.sum(we)
            return pool_cross_entropy
        

        zeros = torch.zeros([m * n]).to(device).float()
        def evaluate_performance(network_output,train_samples_gt,train_samples_gt_onehot, require_AA_KPP=False,printFlag=True):
            if False==require_AA_KPP:
                with torch.no_grad():
                    available_label_idx=(train_samples_gt!=0).float()#有效标签的坐标,用于排除背景
                    available_label_count=available_label_idx.sum()#有效标签的个数
                    correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
                    OA= correct_prediction.cpu()/available_label_count
                    
                    return OA
            else:
                with torch.no_grad():
                    #计算OA
                    available_label_idx=(train_samples_gt!=0).float()#有效标签的坐标,用于排除背景
                    available_label_count=available_label_idx.sum()#有效标签的个数
                    correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
                    OA= correct_prediction.cpu()/available_label_count
                    OA=OA.cpu().numpy()
                    
                    # 计算AA
                    zero_vector = np.zeros([class_count])
                    output_data=network_output.cpu().numpy()
                    train_samples_gt=train_samples_gt.cpu().numpy()
                    train_samples_gt_onehot=train_samples_gt_onehot.cpu().numpy()
                    
                    output_data = np.reshape(output_data, [m * n, class_count])
                    idx = np.argmax(output_data, axis=-1)
                    for z in range(output_data.shape[0]):
                        if ~(zero_vector == output_data[z]).all():
                            idx[z] += 1
                    # idx = idx + train_samples_gt
                    count_perclass = np.zeros([class_count])
                    correct_perclass = np.zeros([class_count])
                    for x in range(len(train_samples_gt)):
                        if train_samples_gt[x] != 0:
                            count_perclass[int(train_samples_gt[x] - 1)] += 1
                            if train_samples_gt[x] == idx[x]:
                                correct_perclass[int(train_samples_gt[x] - 1)] += 1
                    test_AC_list = correct_perclass / count_perclass
                    test_AA = np.average(test_AC_list)

                    # 计算KPP
                    test_pre_label_list = []
                    test_real_label_list = []
                    output_data = np.reshape(output_data, [m * n, class_count])
                    idx = np.argmax(output_data, axis=-1)
                    idx = np.reshape(idx, [m, n])
                    for ii in range(m):
                        for jj in range(n):
                            if Test_GT[ii][jj] != 0:
                                test_pre_label_list.append(idx[ii][jj] + 1)
                                test_real_label_list.append(Test_GT[ii][jj])
                    test_pre_label_list = np.array(test_pre_label_list)
                    test_real_label_list = np.array(test_real_label_list)
                    kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                                      test_real_label_list.astype(np.int16))
                    test_kpp = kappa

                    # 输出
                    if printFlag:
                        print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                        print('acc per class:')
                        print(test_AC_list)

                    OA_ALL.append(OA)
                    AA_ALL.append(test_AA)
                    KPP_ALL.append(test_kpp)
                    AVG_ALL.append(test_AC_list)
                    
                    # 保存数据信息
                    # f = open('C:/Users/Four Eleven/Desktop/code/HCGN/results/'  + dataset_name + '_results.txt', 'a+')
                    f = open('/root/HCGN-main/HCGNresults'  + dataset_name + '_results.txt', 'a+')
                    str_results = '\n======================' \
                                  + " learning rate=" + str(learning_rate) \
                                  + " epochs=" + str(max_epoch) \
                                  + " train ratio=" + str(train_ratio) \
                                  + " val ratio=" + str(val_ratio) \
                                  + " ======================" \
                                  + "\nOA=" + str(OA) \
                                  + "\nAA=" + str(test_AA) \
                                  + '\nkpp=' + str(test_kpp) \
                                  + '\nacc per class:' + str(test_AC_list) + "\n"
                                  # + '\ntrain time:' + str(time_train_end - time_train_start) \
                                  # + '\ntest time:' + str(time_test_end - time_test_start) \
                    f.write(str_results)
                    f.close()
                    return OA
        
        # 训练
        optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)#,weight_decay=0.0001
        best_loss=99999
        net.train()
        # total_num = sum(p.numel() for p in net.parameters())
        # trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        # print('参数量:',total_num,trainable_num)
        # print('Iter=',curr_seed)
        summary(net,(net_input.shape))
        tic1 = time.perf_counter()
        for i in range(max_epoch+1):
            optimizer.zero_grad()  # zero the gradient buffers
            output, CNN_result, GCN_result0,GCN_result1, GCN_result2= net(net_input)
            loss = compute_loss(output,train_samples_gt_onehot,train_label_mask)
            loss.backward(retain_graph=False)
            optimizer.step()  # Does the update
            if i%10==0:
                with torch.no_grad():
                    net.eval()
                    output, CNN_result,GCN_result0,GCN_result1, GCN_result2= net(net_input)
                    trainloss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
                    trainOA = evaluate_performance(output, train_samples_gt, train_samples_gt_onehot)
                    valloss = compute_loss(output, val_samples_gt_onehot, val_label_mask)
                    valOA = evaluate_performance(output, val_samples_gt, val_samples_gt_onehot)
                    print("Epoch={} \t train loss={}\t  train OA={}\t  val loss={}\t  val OA={}".format(str(i + 1), "%.4f" %trainloss, "%.2f%%" % (trainOA * 100), "%.4f" %valloss, "%.2f%%" % (trainOA * 100)))

                    if valloss < best_loss :
                        best_loss = valloss
                        torch.save(net.state_dict(),"/root/HCGN-main/HCGNresults.pth")
                        # print('save model...')
                torch.cuda.empty_cache()
                net.train()
        toc1 = time.perf_counter()
        print("\n\n====================training done. starting evaluation...========================\n")
        training_time=toc1 - tic1 + LDA_SLIC_Time #分割耗时需要算进去
        Train_Time_ALL.append(training_time)
        
        torch.cuda.empty_cache()
        with torch.no_grad():
            net.load_state_dict(torch.load("/root/HCGN-main/HCGNresults.pth"))
            net.eval()
            tic2 = time.perf_counter()
            output, CNN_result,GCN_result0,GCN_result1, GCN_result2 = net(net_input)
            toc2 = time.perf_counter()
            testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask)
            testOA = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot,require_AA_KPP=True,printFlag=False)
            print("{}\ttest loss={}\t test OA={}".format(str(i + 1), testloss, testOA))
            


            classification_map = torch.argmax(output, 1).reshape([height, width]).cpu().numpy()+1
            classification_map[gt == 0] = -1
            gt_classification_map = gt.copy().astype(np.int16)
            gt_classification_map[gt == 0] = -1


             # ===== 新增：t-SNE 可视化并保存到 /userData/gpusjl/IP/T-SNE =====
            try:
                output_dir_tsne = "/root/IP/T-SNE"
                os.makedirs(output_dir_tsne, exist_ok=True)
                softmax_all = output.reshape([height * width, -1]).cpu().numpy()
                test_gt_flat = Test_GT.reshape(-1)
                mask = (test_gt_flat != 0)
                if mask.sum() > 0:
                    features = softmax_all[mask]
                    labels = test_gt_flat[mask].astype(int)
                    save_path_tsne = os.path.join(output_dir_tsne, f"{dataset_name}seed{curr_seed}_tSNE.png")
                    visualize_test_tsne(features, labels, save_path_tsne, dataset_name)
            except Exception as e:
                print("Warning: failed to run/save t-SNE:", e)
            # ===== 新增：打印并保存数值混淆矩阵（终端输出） =====
            output_dir = "/root/IP/Classification_Map/"
            os.makedirs(output_dir, exist_ok=True)
            save_prefix = os.path.join(output_dir, f"{dataset_name}seed{curr_seed}_OA{testOA:.4f}")
            save_and_print_confusion_matrix(pred_map=classification_map,
                                            true_map=gt_classification_map,
                                            class_count=class_count,
                                            save_prefix=save_prefix,
                                            normalize=False,
                                            class_names=None)
            # ==================================================  
            output_dir_gt = "/root/IP/gt/"
            os.makedirs(output_dir_gt, exist_ok=True)
            gt_map_path = os.path.join(output_dir_gt, f"{dataset_name}GT_classification_map")
            Draw_Classification_Map_Bg(
                label=gt_classification_map,
                name=gt_map_path,
                scale=4.0,
                dpi=400,
                color_matrix=np.array([
                        [255, 0, 0],       # Corn - 红色
                        [0, 255, 0],     # Broad-leaf soybean - 绿色
                        [0, 0, 255],       # Sesame - 蓝色
                        [255, 255, 0],       # Cotton - 黄色
                        [173, 216, 230],     # Water - 浅蓝色
                        [255, 0, 255],       # Roads and houses - 紫色
                        [255, 192, 203],   # Rice - 粉色
                        [128, 0, 128],   # Narrow-leaf soybean - 紫红色
                        [0, 100, 0]    # Mixed weed - 深绿色
                    ]), bg_color=[0, 0, 0]  # 背景黑色
                )

            #     [255, 0, 0],       # Corn - 红色
                # [0, 255, 0],     # Broad-leaf soybean - 绿色
                # [0, 0, 255],       # Sesame - 蓝色
                # [255, 255, 0],       # Cotton - 黄色
                # [173, 216, 230],     # Water - 浅蓝色
                # [255, 0, 255],       # Roads and houses - 紫色
                # [255, 192, 203],   # Rice - 粉色
                # [128, 0, 128],   # Narrow-leaf soybean - 紫红色
                # [0, 100, 0]    # Mixed weed - 深绿色
            # ]) 
            # 
            # # #     [147, 67, 46],    # 深棕色
            #         [0, 0, 255],      # 蓝色
            #         [255, 100, 0],    # 橙色
            #         [0, 255, 123],    # 亮绿色
            #         [164, 75, 155],   # 紫色
            #         [101, 174, 255],  # 浅蓝色
            #         [118, 254, 172],  # 薄荷绿
            #         [60, 91, 112],    # 深蓝灰
            
            #         [255, 255, 0],    # 黄色
            #         [255, 255, 125],  # 浅黄色
            #         [255, 0, 255],    # 粉红色
            #         [100, 0, 255],    # 深紫色
            #         [0, 172, 254],    # 天蓝色
            #         [0, 255, 0],      # 绿色
            #         [171, 175, 80],   # 橄榄绿
            #         [101, 193, 60]    # 草绿色
            #         ]),                       
            # # 构造保存路径
            output_dir = "/root/IP/Classification_Map/"
            os.makedirs(output_dir, exist_ok=True)
            map_name = os.path.join(output_dir, f"{dataset_name}seed{curr_seed}_OA{testOA:.4f}")

            # 调用你定义的函数
            Draw_Classification_Map_Bg(
                label=classification_map,
                name=map_name,
                scale=4.0,
                dpi=400,
                color_matrix=np.array([
                    [255, 0, 0],       # Corn - 红色
                    [0, 255, 0],     # Broad-leaf soybean - 绿色
                    [0, 0, 255],       # Sesame - 蓝色
                    [255, 255, 0],       # Cotton - 黄色
                    [173, 216, 230],     # Water - 浅蓝色
                    [255, 0, 255],       # Roads and houses - 紫色
                    [255, 192, 203],   # Rice - 粉色
                    [128, 0, 128],   # Narrow-leaf soybean - 紫红色
                    [0, 100, 0]    # Mixed weed - 深绿色
                ])  # 你可以自定义颜色矩阵，否则用默认
            )
            #计算
            # Classification_map = torch.argmax(output, 1).reshape([height, width]).cpu().numpy()
            # print("classification_map min:", Classification_map.min())
            # print("classification_map max:", Classification_map.max())
            # print("classification_map unique:", np.unique(Classification_map))

            # output_dir = "/userData/gpusjl/IP/Classification_Map/"
            # os.makedirs(output_dir, exist_ok=True)

            # # -------------- 新增：绘制并保存彩色分类图 --------------
            # map_save_path = os.path.join(output_dir,
            #                              f"{dataset_name}seed{curr_seed}_OA{testOA:.4f}.png")
            # Draw_Classification_Map(Classification_map,
            #                         map_save_path,
            #                         dataset_name.rstrip('_'),
            #                         dpi=400)
            # # --------------------------------------------------------

            # sio.savemat(
            #     output_dir + dataset_name + "_classification_map.mat",
            #     {'classification_map': Classification_map}
            # )
            # Draw_Classification_Map(classification_map,"D:/廖帝灵/20221030/GCN/CNN_Enhanced_GCN-master/results/"+dataset_name+str(testOA))
            testing_time=toc2 - tic2 + LDA_SLIC_Time #分割耗时需要算进去
            Test_Time_ALL.append(testing_time)
            ## Saving data
            sio.savemat(dataset_name+"softmax.mat",{'softmax':output.reshape([height*width,-1]).cpu().numpy()})
            # np.save(dataset_name+"A_1", CNN_result.cpu().numpy())
            # np.save(dataset_name+"A_2", A_2.cpu().numpy())

            # # visualize
            # x_visualize = GCN_result0.cpu().numpy() #[1,128,h,w]特征图 
            # # print(data.shape)
            # x_visualize = np.mean(x_visualize,axis=1).reshape(data.shape[0],data.shape[1]) #shape为[h,w]，二维
            # x_visualize = (((x_visualize - np.min(x_visualize))/(np.max(x_visualize)-np.min(x_visualize)))*255).astype(np.uint8) #归一化并映射到0-255的整数，方便伪彩色化   
            # savedir =  'C:/Users/Four Eleven/Desktop/Vis/'              
            # x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)  # 伪彩色处理      
            # cv2.imwrite(savedir+'GCN_result0.jpg',x_visualize) #保存可视化图像

            # x_visualize1 = GCN_result1.cpu().numpy() #[1,128,h,w]特征图 
            # # print(data.shape)
            # x_visualize1 = np.mean(x_visualize1,axis=1).reshape(data.shape[0],data.shape[1]) #shape为[h,w]，二维
            # x_visualize1 = (((x_visualize1 - np.min(x_visualize1))/(np.max(x_visualize1)-np.min(x_visualize1)))*255).astype(np.uint8) #归一化并映射到0-255的整数，方便伪彩色化               
            # x_visualize1 = cv2.applyColorMap(x_visualize1, cv2.COLORMAP_JET)  # 伪彩色处理      
            # cv2.imwrite(savedir+'GCN_result1.jpg',x_visualize1) #保存可视化图像

            # x_visualize2 = GCN_result2.cpu().numpy() #[1,128,h,w]特征图 
            # # print(data.shape)
            # x_visualize2 = np.mean(x_visualize2,axis=1).reshape(data.shape[0],data.shape[1]) #shape为[h,w]，二维
            # x_visualize2 = (((x_visualize2 - np.min(x_visualize2))/(np.max(x_visualize2)-np.min(x_visualize2)))*255).astype(np.uint8) #归一化并映射到0-255的整数，方便伪彩色化               
            # x_visualize2 = cv2.applyColorMap(x_visualize2, cv2.COLORMAP_JET)  # 伪彩色处理      
            # cv2.imwrite(savedir+'GCN_result2.jpg',x_visualize2) #保存可视化图像

        
        torch.cuda.empty_cache()
        del net
        
    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    AVG_ALL = np.array(AVG_ALL)
    Train_Time_ALL=np.array(Train_Time_ALL)
    Test_Time_ALL=np.array(Test_Time_ALL)

    print("\ntrain_ratio={}".format(curr_train_ratio),
          "\n========================================================================")
    print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
    print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
    print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
    print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
    print("Average training time:{}".format(np.mean(Train_Time_ALL)))
    print("Average testing time:{}".format(np.mean(Test_Time_ALL)))
    
    # 保存数据信息
    f = open('/root/HCGN-main/HCGNresults' + dataset_name + '_results.txt', 'a+')
    str_results = '\n\n************************************************' \
    +"\ntrain_ratio={}".format(curr_train_ratio) \
    +'\nOA='+ str(np.mean(OA_ALL))+ '+-'+ str(np.std(OA_ALL)) \
    +'\nAA='+ str(np.mean(AA_ALL))+ '+-'+ str(np.std(AA_ALL)) \
    +'\nKpp='+ str(np.mean(KPP_ALL))+ '+-'+ str(np.std(KPP_ALL)) \
    +'\nAVG='+ str(np.mean(AVG_ALL,0))+ '+-'+ str(np.std(AVG_ALL,0)) \
    +"\nAverage training time:{}".format(np.mean(Train_Time_ALL)) \
    +"\nAverage testing time:{}".format(np.mean(Test_Time_ALL))
    f.write(str_results)
    f.close()
        
    #网络参数数量

def save_and_print_confusion_matrix(pred_map: np.ndarray,
                                    true_map: np.ndarray,
                                    class_count: int,
                                    save_prefix: str,
                                    normalize: bool = False,
                                    class_names: list = None):
    """
    计算、打印并保存混淆矩阵（忽略 true_map 中为 0 的背景像元）
    pred_map / true_map: 2D numpy arrays，标签范围为 1..class_count，背景为0 或 -1
    save_prefix: 保存路径前缀（不含扩展名）
    返回：混淆矩阵（numpy array）
    """
    eps = 1e-12
    # 统一处理 true_map 中的 -1（背景）为 0
    true_map_proc = true_map.copy()
    true_map_proc[true_map_proc < 1] = 0

    mask = (true_map_proc != 0)
    y_true = true_map_proc[mask].ravel()
    y_pred = pred_map[mask].ravel()

    labels = list(range(1, class_count + 1))
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        denom = cm.sum(axis=1, keepdims=True).astype(float)
        cm = cm / (denom + eps)

    # 打印矩阵到控制台
    print("Confusion matrix (rows=true, cols=pred):")
    np.set_printoptions(precision=4, suppress=True)
    print(cm)

    # 保存为 .npy
    np.save(save_prefix + "_confusion.npy", cm)

    # 绘图保存
    plt.figure(figsize=(max(6, class_count * 0.4), max(5, class_count * 0.3)))
    im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_labels = class_names if class_names is not None else labels
    plt.xticks(np.arange(len(tick_labels)), tick_labels, rotation=45)
    plt.yticks(np.arange(len(tick_labels)), tick_labels)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix' + (' (normalized)' if normalize else ''))
    # annotate
    fmt = '.2f' if normalize else 'd'
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(j, i, format(val, fmt),
                     ha="center", va="center",
                     color="white" if val > cm.max() / 2. else "black", fontsize=6)
    plt.tight_layout()
    plt.savefig(save_prefix + ( "_confusion_norm.png" if normalize else "_confusion.png"), dpi=200)
    plt.close()
    return cm










