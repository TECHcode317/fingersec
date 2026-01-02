from google.colab import files
files.upload()


import zipfile, os

with zipfile.ZipFile("fvc2002.zip","r") as z:
    z.extractall("/content/fvc2002")

with zipfile.ZipFile("socofing.zip","r") as z:
    z.extractall("/content/socofing")

FVC_PATH = "/content/fvc2002/fingerprints/DB1_B"
SOCO_PATH = "/content/socofing"

print("FVC2002 samples:", len(os.listdir(FVC_PATH)))


!pip install -q opencv-python scikit-image scikit-learn cryptography numpy matplotlib


import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from skimage.metrics import structural_similarity as ssim


def mobile_preprocess(img):
    img = cv2.resize(img, (160,160))   # mobile display scale
    img = cv2.equalizeHist(img)
    return img


def load_fvc(path):
    imgs=[]
    for f in sorted(os.listdir(path)):
        if f.endswith(".tif"):
            im = cv2.imread(os.path.join(path,f), cv2.IMREAD_GRAYSCALE)
            if im is None: continue
            imgs.append(mobile_preprocess(im))
    return np.array(imgs)

fvc_images = load_fvc(FVC_PATH)
print("Loaded FVC2002 images:", fvc_images.shape)


def load_socofing(base):
    imgs=[]
    for root,_,files in os.walk(base):
        for f in files:
            if f.lower().endswith(".bmp"):
                im = cv2.imread(os.path.join(root,f), cv2.IMREAD_GRAYSCALE)
                if im is None: continue
                imgs.append(mobile_preprocess(im))
    return np.array(imgs)

soc_images = load_socofing(SOCO_PATH)
print("Loaded SOCOFing images:", soc_images.shape)


def lbp_feat(img):
    lbp = local_binary_pattern(img, 8, 1, "uniform")
    hist,_ = np.histogram(lbp.ravel(), bins=np.arange(11), density=True)
    return hist

fvc_feat = np.array([lbp_feat(i) for i in fvc_images])
soc_feat = np.array([lbp_feat(i) for i in soc_images])



X = np.vstack((fvc_feat, soc_feat))
y = np.array([1]*len(fvc_feat) + [0]*len(soc_feat))

Xtr,Xte,ytr,yte = train_test_split(
    X,y,test_size=0.25,random_state=42,stratify=y)

clf = SVC(kernel="rbf", C=5, gamma="scale")
clf.fit(Xtr,ytr)
yp = clf.predict(Xte)

cm = confusion_matrix(yte,yp)
print("\nCONFUSION MATRIX (FVC vs SOCOFing):")
print(cm)


def lattice_random():
    A = np.random.randint(0,12289,(32,32))
    s = np.random.randint(0,12289,(32,1))
    b = (A@s)%12289
    return b.astype(np.uint8).tobytes()


def encrypt(img,pin=b"1234"):
    key = lattice_random()[:32]
    aes = AESGCM(key)
    nonce = os.urandom(12)
    ct = aes.encrypt(nonce,img.tobytes(),pin)
    return ct,nonce,key

def decrypt(ct,nonce,key,pin=b"1234"):
    aes = AESGCM(key)
    pt = aes.decrypt(nonce,ct,pin)
    return np.frombuffer(pt,dtype=np.uint8).reshape(img_shape)


def mse(a,b): return np.mean((a-b)**2)
def psnr(m): return 10*np.log10((255**2)/(m+1e-9))

orig = fvc_images[0]
img_shape = orig.shape

ct,nonce,key = encrypt(orig)
dec = decrypt(ct,nonce,key)


enc_img = np.frombuffer(ct[:orig.size],dtype=np.uint8).reshape(img_shape)

mse_dec = mse(orig,dec)
mse_enc = mse(orig,enc_img)

print("\nIMAGE QUALITY METRICS")
print("Original vs Decrypted → MSE:", round(mse_dec,4),
      "PSNR:", round(psnr(mse_dec),2),
      "SSIM:", round(ssim(orig,dec),4))

print("Original vs Encrypted → MSE:", round(mse_enc,2),
      "PSNR:", round(psnr(mse_enc),2))


plt.figure(figsize=(10,3))
plt.subplot(1,3,1); plt.imshow(orig,cmap="gray"); plt.title("Original")
plt.subplot(1,3,2); plt.imshow(enc_img,cmap="gray"); plt.title("Encrypted")
plt.subplot(1,3,3); plt.imshow(dec,cmap="gray"); plt.title("Decrypted")
plt.axis("off")
plt.show()




