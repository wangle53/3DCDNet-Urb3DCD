import numpy as np
import os
import os.path as osp
import random
import time
import torch
from sklearn.neighbors import KDTree
import open3d as o3d
from collections import OrderedDict
import configs as cfg
tcfg = cfg.CONFIGS['Train']

# np.random.seed(0)

def mkdir(path):
    if not osp.exists(path):
        os.makedirs(path)

def txt2sample(path):
    points = np.loadtxt(path)   
    return points
  

def random_subsample(points, n_samples):
    """
    random subsample points when input points has larger length than n_samples 
    or add zeros for points which has smaller length than n_samples.
    """
    if len(points.shape) == 1:
        points = points.reshape(1, 4)
        
    if points.shape[0]==0:
#         print('No points found at this center replacing with dummy')
        points = np.zeros((n_samples,points.shape[1]))
    if n_samples < points.shape[0]:
        random_indices = np.random.choice(points.shape[0], n_samples, replace=False)
        points = points[random_indices, :]
    if n_samples > points.shape[0]:
        apd = np.zeros((n_samples-points.shape[0], points.shape[1]))
        points = np.vstack((points, apd))
    return points


def align_length(p0_path, p1_path, length):
    """
    output a pair of points with the same length.
    """
    p0 = txt2sample(p0_path) 
    p1 = txt2sample(p1_path)
    p0_raw_length = p0.shape[0]
    p1_raw_length = p1.shape[0]
    if p0.shape[0] != length:
        p0 = random_subsample(p0, length)
    if p1.shape[0] != length:
        p1 = random_subsample(p1, length) 
    return p0, p1, p0_raw_length, p1_raw_length

def get_errors(err):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """
        errors = OrderedDict([
            ('err', err.item()),
            ])

        return errors
        
def plot_current_errors(epoch, counter_ratio, errors,vis):
        """Plot current errros.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            errors (OrderedDict): Error for the current epoch.
        """
        
        plot_data = {}
        plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        plot_data['X'].append(epoch + counter_ratio)
        plot_data['Y'].append([errors[k] for k in plot_data['legend']])
        
        vis.line(win='wire train loss', update='append',
            X = np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
            Y = np.array(plot_data['Y']),
            opts={
                'title': 'Change Detection' + ' loss over time',
                'legend': plot_data['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Loss'
            })

def save_weights(epoch,net,optimizer,save_path, model_name):
    if isinstance(net, torch.nn.DataParallel):
        checkpoint = {
                'epoch': epoch,
                'model_state_dict': net.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
            }
    else:
        checkpoint = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
            }
    torch.save(checkpoint,os.path.join(save_path,'current_%s.pth'%(model_name)))
    if epoch % 1 == 0:
        torch.save(checkpoint,os.path.join(save_path,'%d_%s.pth'%(epoch,model_name)))

def plot_performance( epoch, performance, vis):
        """ Plot performance

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            performance (OrderedDict): Performance for the current epoch.
        """
        plot_res = []
        plot_res = {'X': [], 'Y': [], 'legend': list(performance.keys())}
        plot_res['X'].append(epoch)
        plot_res['Y'].append([performance[k] for k in plot_res['legend']])
        vis.line(win='AUC', update='append',
            X=np.stack([np.array(plot_res['X'])] * len(plot_res['legend']), 1),
            Y=np.array(plot_res['Y']),
            opts={
                'title': 'Testing ' + 'Performance Metrics',
                'legend': plot_res['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Stats'
            },
        ) 

def save_cfg(cfg, path):
    mkdir(path)
    if not cfg.resume:
        if os.path.exists(os.path.join(path, 'configure.txt')):
            os.remove(os.path.join(path, 'configure.txt'))
        if os.path.exists(os.path.join(path, 'train_loss.txt')):
            os.remove(os.path.join(path, 'train_loss.txt'))
        if os.path.exists(os.path.join(path, 'val_metric.txt')):
            os.remove(os.path.join(path, 'val_metric.txt'))
        if os.path.exists(os.path.join(path, 'val_performance.txt')):
            os.remove(os.path.join(path, 'val_performance.txt')) 
        if os.path.exists(os.path.join(path, 'test_performance.txt')):
            os.remove(os.path.join(path, 'test_performance.txt'))      
        with open(os.path.join(path, 'configure.txt'), 'a') as f:
            f.write('---------------{}----------------'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
            f.write('\n')
            f.write('----------------Network and training configure-----------------')
            f.write('\n')
            for k in cfg:
                f.write(str(k)+':')
                f.write(str(cfg[k]))
                f.write('\n')

def save_prediction2(p0, p1, lb0, lb1, scores0, scores1, path, pc0_name, pc1_name, data_path):
    p0_name, p1_name = pc0_name[0], pc1_name[0]
    dir = path.split('\\')[-1]
    raw_dir = dir.split('_')[:-3]
    if len(raw_dir) > 1:
        raw_dir = raw_dir[0] + '_' + raw_dir[1]
    else:
         raw_dir = raw_dir[0]
    path = path.replace(dir, raw_dir)
    mkdir(os.path.join(path, dir))
    p0 = p0[:, :, :3].squeeze(0).detach().cpu().numpy()
    p1 = p1[:, :, :3].squeeze(0).detach().cpu().numpy()
    lb0 = lb0.transpose(1,0).detach().cpu().numpy()
    lb1 = lb1.transpose(1,0).detach().cpu().numpy()
    scores0 = scores0.transpose(1,0).detach().cpu().numpy()
    scores1 = scores1.transpose(1,0).detach().cpu().numpy()
    idx0 = np.where(np.all(p0, axis=1) != 0)[0]  
    idx1 = np.where(np.all(p1, axis=1) != 0)[0]    
    p0 = np.hstack((p0[idx0, :], lb0[idx0, :]*2, scores0[idx0, :]*2))
    p1 = np.hstack((p1[idx1, :], lb1[idx1, :], scores1[idx1, :]))
    
    np.savetxt(osp.join(path, dir,  p0_name), p0, fmt="%.2f %.2f %.2f %.0f %.0f")
    np.savetxt(osp.join(path, dir, p1_name), p1, fmt="%.2f %.2f %.2f %.0f %.0f")    


def search_k_neighbors(raw, query, k):
    search_tree = KDTree(raw)
    _, neigh_idx = search_tree.query(query, k)
    return neigh_idx


def combine_sub_pcs(path):
    '''
    test: LyonS, LyonS1, LyonS2
    val: Lyon2_NorthDown
    '''
    for dir in os.listdir(path):
        pcs1 = []; pcs2 = [];
        for root, _, files in os.walk(os.path.join(path, dir)):
            for file in files:
                if '0' in file:
                    pcs1.append(np.loadtxt(os.path.join(root, file)))
                if '1' in file:
                    pcs2.append(np.loadtxt(os.path.join(root, file)))
        pc1 = np.row_stack(pcs1)
        pc2 = np.row_stack(pcs2)
        if True: # project removed prediction into pc2
            removed_idx = np.where(pc1[:, 4]==2)
            if len(removed_idx[0]) > 0:
                query = pc1[:, :2][removed_idx]
                revised_idx = search_k_neighbors(pc2[:, :2], query, 1)
                pc2[:, 4][revised_idx] = 2
        if os.path.exists(os.path.join(path, dir+'_pointCloud0.txt')):
            os.remove(os.path.join(path, dir+'_pointCloud0.txt'))
        if os.path.exists(os.path.join(path, dir+'_pointCloud1.txt')):
            os.remove(os.path.join(path, dir+'_pointCloud1.txt'))
        np.savetxt(os.path.join(path, dir+'_pointCloud0.txt'), pc1, fmt="%.2f %.2f %.2f %.0f %.0f") 
        np.savetxt(os.path.join(path, dir+'_pointCloud1.txt'), pc2, fmt="%.2f %.2f %.2f %.0f %.0f") 
    print('pfoject label finished')
    
    