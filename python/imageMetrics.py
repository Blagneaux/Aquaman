import pandas as pd
import numpy as np


width, height = 2**6, 2**6

def offset(f):
    # Doesn't change the value so it seems to be useless
    g = f.apply(lambda x: x+100)
    return g

def mse(f,g):
    # f and g are DataFrame
    return f.subtract(g).apply(lambda x: x**2).sum().sum() / (width * height)

def psnr(f,g):
    return 10 * np.log10(255**2 / mse(f,g))

def ssim(f,g):
    mu_f = f.mean().mean()
    mu_g = g.mean().mean()
    L = 255
    c1 = (0.01 * L)**2

    l = (2 * mu_f * mu_g + c1) / (mu_f**2 + mu_g**2 + c1)

    f_array = f.to_numpy()
    g_array = g.to_numpy()
    f_flat = f_array.flatten()
    g_flat = g_array.flatten()

    var_f = np.var(f_flat, dtype=np.float64)
    var_g = np.var(g_flat, dtype=np.float64)
    c2 = (0.03 * L)**2

    c = (2 * np.sqrt(var_f) * np.sqrt(var_g) + c2) / (var_f + var_g + c2)

    cov = np.cov(f_flat, g_flat, bias=True)[0][1]
    c3 = c2 / 2

    s = (cov + c3) / (np.sqrt(var_f) * np.sqrt(var_g) + c3)

    return l * c * s


file_path_original = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_original.csv'
original = pd.read_csv(file_path_original, header=None)
# original = offset(original)

file_path_customShape = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_customShape.csv'
customShape = pd.read_csv(file_path_customShape, header=None)
# customShape = offset(customShape)

file_path_customDhdx = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_customDhdx.csv'
customDhdx = pd.read_csv(file_path_customDhdx, header=None)
# customDhdx = offset(customDhdx)

file_path_customHdot = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_customHdot.csv'
customHdot = pd.read_csv(file_path_customHdot, header=None)
# customHdot = offset(customHdot)

file_path_customDist = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_customDist.csv'
customDist = pd.read_csv(file_path_customDist, header=None)
# customDist = offset(customDist)

file_path_customDhdxHdot = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_customDhdxHdot.csv'
customDhdxHdot = pd.read_csv(file_path_customDhdxHdot, header=None)
# customDhdxHdot = offset(customDhdxHdot)

file_path_customDhdxDist = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_customDhdxDist.csv'
customDhdxDist = pd.read_csv(file_path_customDhdxDist, header=None)
# customDhdxDist = offset(customDhdxDist)

file_path_customHdotDist = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_customHdotDist.csv'
customHdotDist = pd.read_csv(file_path_customHdotDist, header=None)
# customHdotDist = offset(customHdotDist)

file_path_customFull = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_customFull.csv'
customFull = pd.read_csv(file_path_customFull, header=None)
# customFull = offset(customFull)

original_mean = original.mean().mean()
original_max = original.max().max()
original_min = original.min().min()
origianl_l1 = original.apply(lambda x: np.abs(x)).sum().sum()
original_l2 = np.sqrt(original.apply(lambda x: x**2).sum().sum())
original_linf = np.max([original_max, -original_min])

shape_mean = customShape.mean().mean()
shape_max = customShape.max().max()
shape_min = customShape.min().min()
shape_l1 = customShape.apply(lambda x: np.abs(x)).sum().sum()
shape_l2 = np.sqrt(customShape.apply(lambda x: x**2).sum().sum())
shape_linf = np.max([shape_max, -shape_min])

customDhdx_mean = customDhdx.mean().mean()
customDhdx_max = customDhdx.max().max()
customDhdx_min = customDhdx.min().min()
customDhdx_l1 = customDhdx.apply(lambda x: np.abs(x)).sum().sum()
customDhdx_l2 = np.sqrt(customDhdx.apply(lambda x: x**2).sum().sum())
customDhdx_linf = np.max([customDhdx_max, -customDhdx_min])

customHdot_mean = customHdot.mean().mean()
customHdot_max = customHdot.max().max()
customHdot_min = customHdot.min().min()
customHdot_l1 = customHdot.apply(lambda x: np.abs(x)).sum().sum()
customHdot_l2 = np.sqrt(customHdot.apply(lambda x: x**2).sum().sum())
customHdot_linf = np.max([customHdot_max, -customHdot_min])

customDist_mean = customDist.mean().mean()
customDist_max = customDist.max().max()
customDist_min = customDist.min().min()
customDist_l1 = customDist.apply(lambda x: np.abs(x)).sum().sum()
customDist_l2 = np.sqrt(customDist.apply(lambda x: x**2).sum().sum())
customDist_linf = np.max([customDist_max, -customDist_min])

customDhdxHdot_mean = customDhdxHdot.mean().mean()
customDhdxHdot_max = customDhdxHdot.max().max()
customDhdxHdot_min = customDhdxHdot.min().min()
customDhdxHdot_l1 = customDhdxHdot.apply(lambda x: np.abs(x)).sum().sum()
customDhdxHdot_l2 = np.sqrt(customDhdxHdot.apply(lambda x: x**2).sum().sum())
customDhdxHdot_linf = np.max([customDhdxHdot_max, -customDhdxHdot_min])

customDhdxDist_mean = customDhdxDist.mean().mean()
customDhdxDist_max = customDhdxDist.max().max()
customDhdxDist_min = customDhdxDist.min().min()
customDhdxDist_l1 = customDhdxDist.apply(lambda x: np.abs(x)).sum().sum()
customDhdxDist_l2 = np.sqrt(customDhdxDist.apply(lambda x: x**2).sum().sum())
customDhdxDist_linf = np.max([customDhdxDist_max, -customDhdxDist_min])

customHdotDist_mean = customHdotDist.mean().mean()
customHdotDist_max = customHdotDist.max().max()
customHdotDist_min = customHdotDist.min().min()
customHdotDist_l1 = customHdotDist.apply(lambda x: np.abs(x)).sum().sum()
customHdotDist_l2 = np.sqrt(customHdotDist.apply(lambda x: x**2).sum().sum())
customHdotDist_linf = np.max([customHdotDist_max, -customHdotDist_min])

customFull_mean = customFull.mean().mean()
customFull_max = customFull.max().max()
customFull_min = customFull.min().min()
customFull_l1 = customFull.apply(lambda x: np.abs(x)).sum().sum()
customFull_l2 = np.sqrt(customFull.apply(lambda x: x**2).sum().sum())
customFull_linf = np.max([customFull_max, -customFull_min])

print("Original mean, l1 norm, l2 norm and l_inf norm: ", original_mean, origianl_l1, original_l2, original_linf)
print("Custom shape mean, l1 norm, l2 norm and l_inf norm: ", shape_mean, shape_l1, shape_l2, shape_linf)
print("Custom dhdx mean, l1 norm, l2 norm and l_inf norm: ", customDhdx_mean, customDhdx_l1, customDhdx_l2, customDhdx_linf)
print("Custom hdot mean, l1 norm, l2 norm and l_inf norm: ", customHdot_mean, customHdot_l1, customHdot_l2, customHdot_linf)
print("Custom distance mean, l1 norm, l2 norm and l_inf norm: ", customDist_mean, customDist_l1, customDist_l2, customDist_linf)
print("Custom dhdx and hdot mean, l1 norm, l2 norm and l_inf norm: ", customDhdxHdot_mean, customDhdxHdot_l1, customDhdxHdot_l2, customDhdxHdot_linf)
print("Custom dhdx and distance mean, l1 norm, l2 norm and l_inf norm: ", customDhdxDist_mean, customDhdxDist_l1, customDhdxDist_l2, customDhdxDist_linf)
print("Custom hdot and distance mean, l1 norm, l2 norm and l_inf norm: ", customHdotDist_mean, customHdotDist_l1, customHdotDist_l2, customHdotDist_linf)
print("Custom full mean, l1 norm, l2 norm and l_inf norm: ", customFull_mean, customFull_l1, customFull_l2, customFull_linf)

print("------------------------PSNR------------------------")
print("Original: ", psnr(original, original))
print("Shape: ", psnr(original, customShape))
print("dhdx: ", psnr(original, customDhdx))
print("hdot: ", psnr(original, customHdot))
print("distance: ", psnr(original, customDist))
print("dhdx and hdot: ", psnr(original, customDhdxHdot))
print("dhdx and distance: ", psnr(original, customDhdxDist))
print("hdot and distance: ", psnr(original, customHdotDist))
print("full: ", psnr(original, customFull))

allMax = [original_max, shape_max, customDhdx_max, customHdot_max, customDist_max, customDhdxHdot_max, customDhdxDist_max, customHdotDist_max, customFull_max]
allMin = [original_min, shape_min, customDhdx_min, customHdot_min, customDist_min, customDhdxHdot_min, customDhdxDist_min, customHdotDist_min, customFull_min]

print("------------------------SSIM------------------------")
print("Original: ", ssim(original, original))
print("Shape: ", ssim(original, customShape))
print("dhdx: ", ssim(original, customDhdx))
print("hdot: ", ssim(original, customHdot))
print("distance: ", ssim(original, customDist))
print("dhdx and hdot: ", ssim(original, customDhdxHdot))
print("dhdx and distance: ", ssim(original, customDhdxDist))
print("hdot and distance: ", ssim(original, customHdotDist))
print("full: ", ssim(original, customFull))