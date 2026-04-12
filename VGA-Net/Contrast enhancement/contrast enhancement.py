import numpy as np
from scipy.signal import convolve2d
import cv2

# 卷积核的全局变量
kernel_x = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
kernel_y = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])

def l1_l0_minimization(I, mu1, mu2, rho0, max_iterations=100, tolerance=1e-5):
    # 初始化变量
    I_s = np.zeros_like(I)  # 平滑层
    G1 = np.zeros_like(I)  # 平滑层的辅助变量
    G2 = np.zeros_like(I)  # 细节层的辅助变量
    lamb1 = np.zeros_like(I)  # 平滑层的拉格朗日乘子
    lamb2 = np.zeros_like(I)  # 细节层的拉格朗日乘子
    I_s_old = np.zeros_like(I)  # 之前的平滑层
    
    # ADMM 迭代
    for iter in range(max_iterations):
        # 更新平滑层
        I_s = update_smooth_layer(I, I_s, G1, G2, lamb1, rho0, mu1, mu2)
        
        # 更新辅助变量和拉格朗日乘子
        G1, G2, lamb1, lamb2 = update_auxiliary_variables(I, I_s, G1, G2, lamb1, lamb2, rho0)
        
        # 检查收敛性
        diff = np.linalg.norm(I_s - I_s_old) / np.linalg.norm(I_s)
        if diff < tolerance:
            break
        
        I_s_old = I_s
    
    # 计算细节层
    I_detail = I - I_s
    
    return I_detail

def update_smooth_layer(I, I_s, G1, G2, lamb1, rho0, mu1, mu2):
    # 使用 ADMM 更新平滑层
    
    # 计算平滑层的梯度
    grad_I_s_x = convolve2d(I_s, kernel_x, mode='same', boundary='wrap')
    grad_I_s_y = convolve2d(I_s, kernel_y, mode='same', boundary='wrap')
    
    # 使用收缩算子更新平滑层
    term1 = I + grad_I_s_x * rho0
    term2 = I + grad_I_s_y * rho0
    I_s = np.maximum(np.sqrt(term1**2 + term2**2) - mu1 / rho0, 0) * np.sign(I)
    
    return I_s

def update_auxiliary_variables(I, I_s, G1, G2, lamb1, lamb2, rho0):
    # 更新辅助变量和拉格朗日乘子
    
    # 计算梯度
    grad_I_s_x = convolve2d(I_s, kernel_x, mode='same', boundary='wrap')
    grad_I_s_y = convolve2d(I_s, kernel_y, mode='same', boundary='wrap')
    grad_I_x = convolve2d(I, kernel_x, mode='same', boundary='wrap')
    grad_I_y = convolve2d(I, kernel_y, mode='same', boundary='wrap')
    
    # 更新 G1 和 G2
    G1 = grad_I_s_x + lamb1 / rho0
    G2 = grad_I_x - grad_I_s_x + lamb2 / rho0
    
    # 更新拉格朗日乘子
    lamb1 = lamb1 + rho0 * (grad_I_s_x - G1)
    lamb2 = lamb2 + rho0 * ((grad_I_x - grad_I_s_x) - G2)
    
    return G1, G2, lamb1, lamb2



# 加载视网膜图像
retina_image = cv2.imread('Honeyview_im0139.jpg', cv2.IMREAD_GRAYSCALE)

# 设置参数
mu1 = 0.5
mu2 = 0.005
rho0 = 2

# 应用 l1_l0_minimization 算法
enhanced_retina_image = l1_l0_minimization(retina_image, mu1, mu2, rho0)

# 显示原始和增强后的图像
cv2.imshow('Original Retina Image', retina_image)
cv2.imshow('Honeyview_im0139_enhanced', enhanced_retina_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
