import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np

data = scio.loadmat('testData.mat')
testData_raw = data['rawData']
testData = data['rawData'].reshape(256*256, 8).transpose()
noiseData = data['noiseData'].reshape(256*256, 8).transpose()
channels = testData.shape[0]

# norm_noiseData = np.zeros((8, 256*256))
# for i in range(8):
#     x = np.real(np.fft.ifft2(noiseData[i].reshape(256, 256).T).reshape(-1))
#     norm_noiseData[i] = (x - np.mean(x)) / np.std(x)
# print("norm_noise_data:", norm_noiseData)


cor_list = np.zeros((channels, channels))
for i in range(channels):
    for j in range(channels):
        x = np.abs(noiseData[i])
        y = np.abs(noiseData[j])
        # cor_list[i, j] = (np.dot(x, y.T) / x.shape[0] - np.mean(x) * np.mean(y)) / np.std(x) * np.std(y)
        cor_list[i, j] = np.correlate(
            np.abs(noiseData[i]), np.abs(noiseData[j]))
        # cor_list[i, j] = np.correlate(x, y)

# cor_list = np.corrcoef(np.abs(noiseData))
# cor_list = np.matmul(np.abs(noiseData), np.abs(noiseData.T)) / (256 * 256)

print("corr_list:", cor_list)

# cor = np.matmul(noiseData, noiseData.transpose())
eigenvalue, featurevector = np.linalg.eig(cor_list)
print("eigenvalue:", eigenvalue, "featurevector:", featurevector)

print("PAP:", np.matmul(np.matmul(featurevector, cor_list), featurevector.T))
S = np.dot(featurevector, testData).reshape(-1, 256, 256).transpose(1, 2, 0)

ishift_array_raw = np.zeros(S.shape)
ishift_array = np.zeros(S.shape)
for i in range(channels):
    io_raw = np.fft.ifft2(testData_raw[:, :, i])
    io_raw = np.abs(io_raw)  # 逆傅里叶变换
    ishift_array_raw[:, :, i] = io_raw
    io = np.fft.ifft2(S[:, :, i])
    io = np.abs(io)  # 逆傅里叶变换
    ishift_array[:, :, i] = io

raw_picture = np.sqrt(np.sum(np.square(ishift_array_raw), axis=-1))
sos_picture = np.sqrt(np.sum(np.square(ishift_array), axis=-1))

plt.subplot(131)
# plt.imshow(np.matmul(featurevector, cor_list), cmap='gray')
plt.imshow(cor_list, cmap='gray')
plt.title('cor_list')
plt.axis('off')
plt.subplot(132)
plt.imshow(np.fft.fftshift(raw_picture), cmap='gray')
plt.title('raw_picture')
plt.axis('off')
plt.subplot(133)
plt.imshow(np.fft.fftshift(sos_picture), cmap='gray')
plt.title('s_picture')
plt.axis('off')
plt.show()
