!pip install cryptography opencv-python-headless --quiet

import os, cv2, numpy as np, hashlib, secrets
import matplotlib.pyplot as plt
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sklearn.metrics import confusion_matrix, roc_curve, auc


FVC_PATH   = "/content/FVC2002_DB1"
SOCO_PATH  = "/content/SOCOFing/Altered/Scar"  
PIN = "246810"
PBKDF2_ITERS = 15000
BACKEND = default_backend()
TEMPLATE_SIZE = (96,96)


def load_fvc(path):
    images, labels = [], []
    for sid in sorted(os.listdir(path)):
        sp = os.path.join(path, sid)
        if not os.path.isdir(sp): continue
        for fn in os.listdir(sp):
            if fn.lower().endswith(('.tif','.bmp','.png','.jpg')):
                images.append(os.path.join(sp,fn))
                labels.append(sid)
    return images, labels

def load_soc(path):
    imgs = []
    for fn in os.listdir(path):
        if fn.lower().endswith(('.png','.jpg','.bmp')):
            imgs.append(os.path.join(path,fn))
    return imgs


def extract_features(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, TEMPLATE_SIZE)
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (5,5), 1.2)
    return img.flatten().astype(np.float32)/255.0


def lwe_salt():
    A = np.random.randint(0,256,(8,8))
    s = np.random.randint(0,4,8)
    b = A.dot(s)
    return hashlib.sha256(A.tobytes()+b.tobytes()).digest()[:16]

def derive_key(pin,salt):
    kdf = PBKDF2HMAC(hashes.SHA512(),32,salt,PBKDF2_ITERS,BACKEND)
    return kdf.derive(pin.encode())

def encrypt(key,data):
    nonce = secrets.token_bytes(12)
    ct = AESGCM(key).encrypt(nonce,data,None)
    return nonce, ct

def decrypt(key,nonce,ct):
    return AESGCM(key).decrypt(nonce,ct,None)


def match(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)


fvc_imgs, fvc_labels = load_fvc(FVC_PATH)


enroll_imgs = fvc_imgs[::2]
enroll_lbls = fvc_labels[::2]
test_imgs   = fvc_imgs[1::2]
test_lbls   = fvc_labels[1::2]


enrolled=[]
for p,l in zip(enroll_imgs,enroll_lbls):
    feat = extract_features(p)
    salt = lwe_salt()
    key  = derive_key(PIN,salt)
    nonce,ct = encrypt(key, feat.tobytes())
    enrolled.append((l,nonce,ct,salt))


y_true=[]
y_score=[]

for p,l in zip(test_imgs,test_lbls):
    probe = extract_features(p)

    
    g=[]
    for l2,n,ct,s in enrolled:
        if l2==l:
            dec = decrypt(derive_key(PIN,s),n,ct)
            g.append(match(probe,np.frombuffer(dec,np.float32)))
    y_true.append(1)
    y_score.append(max(g))

    
    i=[]
    for l2,n,ct,s in enrolled:
        if l2!=l:
            dec = decrypt(derive_key(PIN,s),n,ct)
            i.append(match(probe,np.frombuffer(dec,np.float32)))
    y_true.append(0)
    y_score.append(np.mean(i))


TH = np.mean(y_score)
y_pred = (np.array(y_score)>=TH).astype(int)
cm_fvc = confusion_matrix(y_true,y_pred)

print("\n CONFUSION MATRIX")
print(cm_fvc)

# ROC
fpr,tpr,_ = roc_curve(y_true,y_score)
roc_auc = auc(fpr,tpr)

plt.figure()
plt.imshow(cm_fvc)
for i in range(2):
    for j in range(2):
        plt.text(j,i,cm_fvc[i,j],ha="center",va="center")
plt.title("Confusion Matrix")
plt.show()

plt.figure()
plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--')
plt.title("ROC"); plt.grid(); plt.show()

soc_imgs = load_soc(SOCO_PATH)

y_true_ab=[]
y_score_ab=[]

for p in soc_imgs[:len(test_imgs)]:
    probe = extract_features(p)

    g=[]
    for l2,n,ct,s in enrolled:
        dec = decrypt(derive_key(PIN,s),n,ct)
        g.append(match(probe,np.frombuffer(dec,np.float32)))

    y_true_ab.append(0)              
    y_score_ab.append(max(g))

TH2 = np.mean(y_score_ab)
y_pred_ab = (np.array(y_score_ab)>=TH2).astype(int)
cm_soc = confusion_matrix(y_true_ab,y_pred_ab)

print("\n SOCOFing ABLATION CONFUSION MATRIX")
print(cm_soc)

plt.figure()
plt.imshow(cm_soc)
for i in range(2):
    for j in range(2):
        plt.text(j,i,cm_soc[i,j],ha="center",va="center")
plt.title("SOCOFing Ablation Confusion Matrix")
plt.show()

print("\n Proper FVC main test + SOCOFing ablation completed")
