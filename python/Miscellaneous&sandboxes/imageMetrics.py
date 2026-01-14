import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


width, height = 2**6, 2**6

# Define the metric functions to be compared ------------------------------------------------------------------
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


# Load all the maps for every combinaison of customization----------------------------------------------------
file_path_original = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_original.csv'
original = pd.read_csv(file_path_original, header=None)
# original = offset(original)
mean_original_field = original.mean(axis=1, skipna=True)
mean_original_field.to_csv('mean_original_field.csv', index=False, header=None)

file_path_customShape = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_customShape.csv'
customShape = pd.read_csv(file_path_customShape, header=None)
# customShape = offset(customShape)
mean_customShape_field = customShape.mean(axis=1, skipna=True)
mean_customShape_field.to_csv('mean_customShape_field.csv', index=False, header=None)

file_path_customDhdx = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_customDhdx.csv'
customDhdx = pd.read_csv(file_path_customDhdx, header=None)
# customDhdx = offset(customDhdx)
mean_customDhdx_field = customDhdx.mean(axis=1, skipna=True)
mean_customDhdx_field.to_csv("mean_customDhdx_field.csv", index=False, header=None)

file_path_customHdot = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_customHdot.csv'
customHdot = pd.read_csv(file_path_customHdot, header=None)
# customHdot = offset(customHdot)
mean_customHdot_field = customHdot.mean(axis=1, skipna=True)
mean_customHdot_field.to_csv("mean_customHdot_field.csv", index=False, header=None)

file_path_customDist = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_customDist.csv'
customDist = pd.read_csv(file_path_customDist, header=None)
# customDist = offset(customDist)
mean_customDist_field = customDist.mean(axis=1, skipna=True)
mean_customDist_field.to_csv("mean_customDist_field.csv", index=False, header=None)

file_path_customDhdxHdot = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_customDhdxHdot.csv'
customDhdxHdot = pd.read_csv(file_path_customDhdxHdot, header=None)
# customDhdxHdot = offset(customDhdxHdot)
mean_customDhdxHdot_field = customDhdxHdot.mean(axis=1, skipna=True)
mean_customDhdxHdot_field.to_csv("mean_customDhdxHdot_field.csv", index=False, header=None)

file_path_customDhdxDist = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_customDhdxDist.csv'
customDhdxDist = pd.read_csv(file_path_customDhdxDist, header=None)
# customDhdxDist = offset(customDhdxDist)
mean_customDhdxDist_field = customDhdxDist.mean(axis=1, skipna=True)
mean_customDhdxDist_field.to_csv("mean_customDhdxDist_field.csv", index=False, header=None)

file_path_customHdotDist = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_customHdotDist.csv'
customHdotDist = pd.read_csv(file_path_customHdotDist, header=None)
# customHdotDist = offset(customHdotDist)
mean_customHdotDist_field = customHdotDist.mean(axis=1, skipna=True)
mean_customHdotDist_field.to_csv("mean_customHdotDist_field.csv", index=False, header=None)

file_path_customFull = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_customFull.csv'
customFull = pd.read_csv(file_path_customFull, header=None)
# customFull = offset(customFull)
mean_customFull_field = customFull.mean(axis=1, skipna=True)
mean_customFull_field.to_csv("mean_customFull_field.csv", index=False, header=None)

file_path_customOpen = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_CustomOpen.csv'
customOpen = pd.read_csv(file_path_customOpen, header=None)
# customOpen = offset(customOpen)
mean_customOpen_field = customOpen.mean(axis=1, skipna=True)
customOpen_field = customOpen.sub(mean_customOpen_field, axis=1)
rms_customOpen_field = pd.DataFrame(np.sqrt(customOpen_field.apply(lambda x: x**2).sum(axis=1)))
mean_customOpen_field.to_csv("mean_customOpen_field.csv", index=False, header=None)
rms_customOpen_field.to_csv("rms_customOpen_field.csv", index=False, header=None)

file_path_originalOpen = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_OriginalOpen.csv'
originalOpen = pd.read_csv(file_path_originalOpen, header=None)
# originalOpen = offset(originalOpen)
mean_originalOpen_field = originalOpen.mean(axis=1, skipna=True)
originalOpen_field = originalOpen.sub(mean_originalOpen_field, axis=1)
rms_originalOpen_field = pd.DataFrame(np.sqrt(originalOpen_field.apply(lambda x: x**2).sum(axis=1)))
mean_originalOpen_field.to_csv("mean_originalOpen_field.csv", index=False, header=None)
rms_originalOpen_field.to_csv("rms_originalOpen_field.csv", index=False, header=None)

# file_path_customTest = 'C:/Users/blagn771/Documents/Aquaman/Aquaman/lily-pad-master/LilyPad/testDataSave/pressure_map_test.csv'
# customTest = pd.read_csv(file_path_customTest, header=None)
# # customTest = offset(customTest)
# mean_customTest_field = customTest.mean(axis=1, skipna=True)
# test_field = customTest.sub(mean_customTest_field, axis=1)
# rms_test_field = pd.DataFrame(np.sqrt(test_field.apply(lambda x: x**2).sum(axis=1)))
# mean_customTest_field.to_csv("mean_customTest_field.csv", index=False, header=None)
# rms_test_field.to_csv("rms_test_field.csv", index=False, header=None)

# Compute the absolute metrics for every combinaison of customization ----------------------------------------
original_mean = original.mean().mean()
original_max = original.max().max()
original_min = original.min().min()
origianl_l1 = original.apply(lambda x: np.abs(x)).sum().sum()
original_l2 = np.sqrt(original.apply(lambda x: x**2).sum().sum())
original_linf = np.max([original_max, -original_min])
original_array = original.to_numpy()
original_flat = original_array.flatten()
var_original = np.var(original_flat, dtype=np.float64)

shape_mean = customShape.mean().mean()
shape_max = customShape.max().max()
shape_min = customShape.min().min()
shape_l1 = customShape.apply(lambda x: np.abs(x)).sum().sum()
shape_l2 = np.sqrt(customShape.apply(lambda x: x**2).sum().sum())
shape_linf = np.max([shape_max, -shape_min])
shape_array = customShape.to_numpy()
shape_flat = shape_array.flatten()
var_shape = np.var(shape_flat, dtype=np.float64)

customDhdx_mean = customDhdx.mean().mean()
customDhdx_max = customDhdx.max().max()
customDhdx_min = customDhdx.min().min()
customDhdx_l1 = customDhdx.apply(lambda x: np.abs(x)).sum().sum()
customDhdx_l2 = np.sqrt(customDhdx.apply(lambda x: x**2).sum().sum())
customDhdx_linf = np.max([customDhdx_max, -customDhdx_min])
customDhdx_array = customDhdx.to_numpy()
customDhdx_flat = customDhdx_array.flatten()
var_customDhdx = np.var(customDhdx_flat, dtype=np.float64)

customHdot_mean = customHdot.mean().mean()
customHdot_max = customHdot.max().max()
customHdot_min = customHdot.min().min()
customHdot_l1 = customHdot.apply(lambda x: np.abs(x)).sum().sum()
customHdot_l2 = np.sqrt(customHdot.apply(lambda x: x**2).sum().sum())
customHdot_linf = np.max([customHdot_max, -customHdot_min])
customHdot_array = customHdot.to_numpy()
customHdot_flat = customHdot_array.flatten()
var_customHdot = np.var(customHdot_flat, dtype=np.float64)

customDist_mean = customDist.mean().mean()
customDist_max = customDist.max().max()
customDist_min = customDist.min().min()
customDist_l1 = customDist.apply(lambda x: np.abs(x)).sum().sum()
customDist_l2 = np.sqrt(customDist.apply(lambda x: x**2).sum().sum())
customDist_linf = np.max([customDist_max, -customDist_min])
customDist_array = customDist.to_numpy()
customDist_flat = customDist_array.flatten()
var_customDist = np.var(customDist_flat, dtype=np.float64)

customDhdxHdot_mean = customDhdxHdot.mean().mean()
customDhdxHdot_max = customDhdxHdot.max().max()
customDhdxHdot_min = customDhdxHdot.min().min()
customDhdxHdot_l1 = customDhdxHdot.apply(lambda x: np.abs(x)).sum().sum()
customDhdxHdot_l2 = np.sqrt(customDhdxHdot.apply(lambda x: x**2).sum().sum())
customDhdxHdot_linf = np.max([customDhdxHdot_max, -customDhdxHdot_min])
customDhdxHdot_array = customDhdxHdot.to_numpy()
customDhdxHdot_flat = customDhdxHdot_array.flatten()
var_customDhdxHdot = np.var(customDhdxHdot_flat, dtype=np.float64)

customDhdxDist_mean = customDhdxDist.mean().mean()
customDhdxDist_max = customDhdxDist.max().max()
customDhdxDist_min = customDhdxDist.min().min()
customDhdxDist_l1 = customDhdxDist.apply(lambda x: np.abs(x)).sum().sum()
customDhdxDist_l2 = np.sqrt(customDhdxDist.apply(lambda x: x**2).sum().sum())
customDhdxDist_linf = np.max([customDhdxDist_max, -customDhdxDist_min])
customDhdxDist_array = customDhdxDist.to_numpy()
customDhdxDist_flat = customDhdxDist_array.flatten()
var_customDhdxDist = np.var(customDhdxDist_flat, dtype=np.float64)

customHdotDist_mean = customHdotDist.mean().mean()
customHdotDist_max = customHdotDist.max().max()
customHdotDist_min = customHdotDist.min().min()
customHdotDist_l1 = customHdotDist.apply(lambda x: np.abs(x)).sum().sum()
customHdotDist_l2 = np.sqrt(customHdotDist.apply(lambda x: x**2).sum().sum())
customHdotDist_linf = np.max([customHdotDist_max, -customHdotDist_min])
customHdotDist_array = customHdotDist.to_numpy()
customHdotDist_flat = customHdotDist_array.flatten()
var_customHdotDist = np.var(customHdotDist_flat, dtype=np.float64)

customFull_mean = customFull.mean().mean()
customFull_max = customFull.max().max()
customFull_min = customFull.min().min()
customFull_l1 = customFull.apply(lambda x: np.abs(x)).sum().sum()
customFull_l2 = np.sqrt(customFull.apply(lambda x: x**2).sum().sum())
customFull_linf = np.max([customFull_max, -customFull_min])
customFull_array = customFull.to_numpy()
customFull_flat = customFull_array.flatten()
var_customFull = np.var(customFull_flat, dtype=np.float64)

customOpen_mean = customOpen.mean().mean()
customOpen_max = customOpen.max().max()
customOpen_min = customOpen.min().min()
customOpen_l1 = customOpen.apply(lambda x: np.abs(x)).sum().sum()
customOpen_l2 = np.sqrt(customOpen.apply(lambda x: x**2).sum().sum())
customOpen_linf = np.max([customOpen_max, -customOpen_min])
customOpen_array = customOpen.to_numpy()
customOpen_flat = customOpen_array.flatten()
var_customOpen = np.var(customOpen_flat, dtype=np.float64)

originalOpen_mean = originalOpen.mean().mean()
originalOpen_max = originalOpen.max().max()
originalOpen_min = originalOpen.min().min()
originalOpen_l1 = originalOpen.apply(lambda x: np.abs(x)).sum().sum()
originalOpen_l2 = np.sqrt(originalOpen.apply(lambda x: x**2).sum().sum())
originalOpen_linf = np.max([originalOpen_max, -originalOpen_min])
originalOpen_array = originalOpen.to_numpy()
originalOpen_flat = originalOpen_array.flatten()
var_originalOpen = np.var(originalOpen_flat, dtype=np.float64)

display = True

if display:
    # Print all the metrics --------------------------------------------------------------------------------------
    print("Original mean, l1 norm, l2 norm and l_inf norm: ", original_mean, origianl_l1, original_l2, original_linf, var_original)
    print("Custom shape mean, l1 norm, l2 norm and l_inf norm: ", shape_mean, shape_l1, shape_l2, shape_linf, var_shape)
    print("Custom dhdx mean, l1 norm, l2 norm and l_inf norm: ", customDhdx_mean, customDhdx_l1, customDhdx_l2, customDhdx_linf, var_customDhdx)
    print("Custom hdot mean, l1 norm, l2 norm and l_inf norm: ", customHdot_mean, customHdot_l1, customHdot_l2, customHdot_linf, var_customHdot)
    print("Custom distance mean, l1 norm, l2 norm and l_inf norm: ", customDist_mean, customDist_l1, customDist_l2, customDist_linf, var_customDist)
    print("Custom dhdx and hdot mean, l1 norm, l2 norm and l_inf norm: ", customDhdxHdot_mean, customDhdxHdot_l1, customDhdxHdot_l2, customDhdxHdot_linf, var_customDhdxHdot)
    print("Custom dhdx and distance mean, l1 norm, l2 norm and l_inf norm: ", customDhdxDist_mean, customDhdxDist_l1, customDhdxDist_l2, customDhdxDist_linf, var_customDhdxDist)
    print("Custom hdot and distance mean, l1 norm, l2 norm and l_inf norm: ", customHdotDist_mean, customHdotDist_l1, customHdotDist_l2, customHdotDist_linf, var_customHdotDist)
    print("Custom full mean, l1 norm, l2 norm and l_inf norm: ", customFull_mean, customFull_l1, customFull_l2, customFull_linf, var_customFull)
    print("Custom full open mean, l1 norm, l2 norm and l_inf norm: ", customOpen_mean, customOpen_l1, customOpen_l2, customOpen_linf, var_customOpen)
    print("Original full open mean, l1 norm, l2 norm and l_inf norm: ", originalOpen_mean, originalOpen_l1, originalOpen_l2, originalOpen_linf, var_originalOpen)

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

    # Create plots for the absolute metrics
    labels = ['Original', 'Custom shape', 'Custom dhdx', 'Custom hdot', 'Custom dist', 'Custom dhdx \n and hdot', 'Custom dhdx \n and dist', 'Custom hdot \n and dist', 'Full custom']
    meanP = [original_mean, shape_mean, customDhdx_mean, customHdot_mean, customDist_mean, customDhdxHdot_mean, customDhdxDist_mean, customHdotDist_mean, customFull_mean]
    meanP = [x * 100 / original_mean for x in meanP]
    l1P = [origianl_l1, shape_l1, customDhdx_l1, customHdot_l1, customDist_l1, customDhdxHdot_l1, customDhdxDist_l1, customHdotDist_l1, customFull_l1]
    l1P = [x * 100 / origianl_l1 for x in l1P]
    l2P = [original_l2, shape_l2, customDhdx_l2, customHdot_l2, customDist_l2, customDhdxHdot_l2, customDhdxDist_l2, customHdotDist_l2, customFull_l2]
    l2P = [x * 100 / original_l2 for x in l2P]
    linfP = [original_linf, shape_linf, customDhdx_linf, customHdot_linf, customDist_linf, customDhdxHdot_linf, customDhdxDist_linf, customHdotDist_linf, customFull_linf]
    linfP = [x * 100 / original_linf for x in linfP]
    var = [var_original, var_shape, var_customDhdx, var_customHdot, var_customDist, var_customDhdxHdot, var_customDhdxDist, var_customHdotDist, var_customFull]
    var = [x * 100 / var_original for x in var]
    PSNR = [np.NAN, 100, 99.2, 73.4, 94.9, 73.1, 94.6, 80.3, 80.3]
    SSIM = [100, 99.97, 99.97, 99.7, 99.96, 99.7, 99.96, 99.8, 99.8]

    fig, ax1 = plt.subplots()
    ax1.plot(labels, meanP)
    ax1.plot(labels, l1P)
    ax1.plot(labels, l2P)
    plt.legend(['mean', 'l1 norm', 'l2 norm'], loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(labels, linfP, 'r')
    ax2. plot(labels, var, 'y')
    plt.legend(['linf norm', 'variance'], loc='upper right')
    plt.xticks(labels, labels, rotation=0, wrap=True)
    plt.title('Pressure metrics')
    plt.show()

    fig, ax1 = plt.subplots()
    ax1.plot(labels, PSNR)
    plt.legend(['PSNR'], loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(labels, SSIM, 'r')
    plt.legend(['SSIM'])
    plt.xticks(labels, labels, rotation=0, wrap=True)
    plt.title('Pressure metrics')
    plt.show()
