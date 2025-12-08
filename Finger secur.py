import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage import img_as_ubyte
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import hashlib
import secrets


FVC_PATH = "/content/FVC2002_DB1"   
SOCOFING_PATH = "/content/SOCOFing" 
IMAGE_SIZE = (256, 256)            
TEMPLATE_BYTES = 512               
PBKDF2_ITERS = 100000              
BACKEND = default_backend()

def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.resize(img, IMAGE_SIZE)
    return img

def normalize(img, mean=128, var=100):
    img = img.astype(np.float32)
    m = img.mean()
    v = img.var()
  
    if v < 1e-6:
        return img.astype(np.uint8)
    res = mean + np.sqrt((var * (img - m)**2) / v) * np.sign(img - m)
    res = np.clip(res, 0, 255).astype(np.uint8)
    return res

def gabor_enhance(img):
  
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    accum = np.zeros_like(img, dtype=np.float32)
    for theta in thetas:
        kernel = cv2.getGaborKernel((21,21), 4.0, theta, 8.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(img.astype(np.float32), cv2.CV_32F, kernel)
        accum = np.maximum(accum, filtered)
    accum = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX)
    return accum.astype(np.uint8)

def binarize_thin(img):

    th = threshold_otsu(img)
    bw = (img > th).astype(np.uint8)
    
    bw = 1 - bw
    skeleton = skeletonize(bw).astype(np.uint8)
    return skeleton


def extract_minutiae_from_skeleton(skel):
  
    minutiae = []
    h, w = skel.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skel[y, x] != 1:
                continue
        
            P = [skel[y, x+1], skel[y-1, x+1], skel[y-1, x], skel[y-1, x-1],
                 skel[y, x-1], skel[y+1, x-1], skel[y+1, x], skel[y+1, x+1], skel[y, x+1]]
            cn = 0
            for i in range(0,8):
                cn += abs(P[i] - P[i+1])
            cn = cn / 2
            if cn == 1:  
                
                theta = estimate_orientation(skel, x, y)
                minutiae.append((x, y, theta, 'ending'))
            elif cn == 3: 
                theta = estimate_orientation(skel, x, y)
                minutiae.append((x, y, theta, 'bifurcation'))
   
    minutiae = filter_close_minutiae(minutiae, min_dist=8)
    return minutiae

def estimate_orientation(skel, x, y, window=7):
    h, w = skel.shape
    xs = []
    ys = []
    for dy in range(-1,2):
        for dx in range(-1,2):
            nx, ny = x+dx, y+dy
            if 0 <= nx < w and 0 <= ny < h and skel[ny, nx]==1:
                xs.append(dx)
                ys.append(dy)
    if len(xs) < 1:
        return 0.0
    vx = sum(xs)
    vy = sum(ys)
    angle = math.atan2(vy, vx) 
    return angle

def filter_close_minutiae(mins, min_dist=6):
    if not mins:
        return mins
    kept = []
    for m in mins:
        x,y,th,t = m
        too_close = False
        for k in kept:
            kx, ky, _, _ = k
            if (x - kx)**2 + (y - ky)**2 < min_dist**2:
                too_close = True
                break
        if not too_close:
            kept.append(m)
    return kept

def encode_minutiae_template(minutiae, max_minutiae=60):
   
    vec = np.zeros((max_minutiae, 3), dtype=np.float32)
    h, w = IMAGE_SIZE[1], IMAGE_SIZE[0] 
    for i, m in enumerate(minutiae[:max_minutiae]):
        x,y,theta,_ = m
        vec[i,0] = float(x) / w
        vec[i,1] = float(y) / h
        vec[i,2] = (theta + math.pi) / (2*math.pi) 
    flat = vec.flatten()
    b = (flat.astype(np.float32)).tobytes()
    if len(b) > TEMPLATE_BYTES:
        return b[:TEMPLATE_BYTES]
    else:
        return b + b'\x00' * (TEMPLATE_BYTES - len(b))

def decode_minutiae_template(b, max_minutiae=60):
    arr = np.frombuffer(b[:max_minutiae*3*4], dtype=np.float32)
    arr = arr.reshape((max_minutiae,3))
    mins = []
    h, w = IMAGE_SIZE[1], IMAGE_SIZE[0]
    for i in range(max_minutiae):
        x = arr[i,0] * w
        y = arr[i,1] * h
        theta = arr[i,2] * 2*math.pi - math.pi
       
        if arr[i].sum() != 0:
            mins.append((x,y,theta))
    return mins

def lwe_randomness(seed=None, n=32, q=4096):
  
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(secrets.randbits(32))
    A = rng.randint(0, q, size=(n,n), dtype=np.int64)
    s = rng.randint(0, 4, size=(n,), dtype=np.int64)       
    e = rng.normal(0, 1.0, size=(n,)).astype(np.int64)     
    b = (A.dot(s) + e) % q
    m = hashlib.sha512()
    m.update(A.tobytes())
    m.update(b.tobytes())
    m.update(s.tobytes())
    rand = m.digest()[:32]
    return rand  # 32 bytes


def derive_key_from_pin(pin, salt, iterations=PBKDF2_ITERS):
    pin_bytes = pin.encode('utf-8')
   
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA512(),
        length=32,
        salt=salt,
        iterations=iterations,
        backend=BACKEND
    )
    key = kdf.derive(pin_bytes)
    return key

def encrypt_template(aes_key, plaintext, nonce=None):
    if nonce is None:
        nonce = secrets.token_bytes(12)
    aesgcm = AESGCM(aes_key)
    ct = aesgcm.encrypt(nonce, plaintext, None)
   
    return nonce, ct 

def decrypt_template(aes_key, nonce, ct):
    aesgcm = AESGCM(aes_key)
    pt = aesgcm.decrypt(nonce, ct, None)
    return pt

def minutiae_match_score(minsA, minsB, dist_thresh=12, angle_thresh=0.35):
    if len(minsA) == 0 or len(minsB) == 0:
        return 0.0
    usedB = set()
    matches = 0
    for ax,ay,ath in minsA:
        best = None
        for j,(bx,by,bth) in enumerate(minsB):
            if j in usedB: 
                continue
            d = math.hypot(ax-bx, ay-by)
            da = min(abs(ath-bth), 2*math.pi - abs(ath-bth))
            if d <= dist_thresh and da <= angle_thresh:
                best = j
                break
        if best is not None:
            matches += 1
            usedB.add(best)
  
    denom = max(1, (len(minsA)+len(minsB))/2.0)
    return float(matches) / denom

def process_image_to_template_bytes(img_path):
    img = read_gray(img_path)
    img_n = normalize(img)
    img_e = gabor_enhance(img_n)
    skel = binarize_thin(img_e)
    mins = extract_minutiae_from_skeleton(skel)
    tbytes = encode_minutiae_template(mins)
    return tbytes, mins

def load_subject_templates(base_path, max_images_per_subject=None):
    templates = []  
    for subj in sorted(os.listdir(base_path)):
        subj_path = os.path.join(base_path, subj)
        if not os.path.isdir(subj_path): continue
        try:
            sid = int(''.join(filter(str.isdigit, subj)))
        except:
            sid = hash(subj) & 0xffffffff
        files = sorted([f for f in os.listdir(subj_path) if f.lower().endswith(('.png','.tif','.jpg','.bmp'))])
        if max_images_per_subject is not None:
            files = files[:max_images_per_subject]
        for f in files:
            path = os.path.join(subj_path, f)
            tbytes, mins = process_image_to_template_bytes(path)
            templates.append((sid, tbytes, mins, path))
    return templates


def enroll_templates(templates, pin="123456"):
    enrolled = []
    for sid, tbytes, mins, path in templates:
  
        lwe_rand = lwe_randomness()
        salt = hashlib.sha256(lwe_rand + b'salt').digest()[:16]
        nonce_seed = hashlib.sha256(lwe_rand + b'nonce').digest()[:12] 
   
        key = derive_key_from_pin(pin, salt)
        nonce = nonce_seed
        nonce, ct = encrypt_template(key, tbytes, nonce=nonce)
        rec = {"sid": sid, "ciphertext": ct, "nonce": nonce, "salt": salt, "path": path}
        enrolled.append(rec)
    return enrolled

def verify_one(enrolled_record, live_template_bytes, live_mins, pin="123456", score_threshold=0.28):
    salt = enrolled_record["salt"]
    nonce = enrolled_record["nonce"]
    ciphertext = enrolled_record["ciphertext"]
    try:
        key = derive_key_from_pin(pin, salt)
        dec = decrypt_template(key, nonce, ciphertext)
    except Exception as e:
       
        return 0.0
  
    enrolled_mins = decode_minutiae_template(dec)
    score = minutiae_match_score(enrolled_mins, live_mins)
    return score

print("Loading templates from FVC folder (this can take time)...")
if not os.path.exists(FVC_PATH):
    print("FVC_PATH not found. Upload dataset to", FVC_PATH)
else:
    fvc_templates = load_subject_templates(FVC_PATH)
    print("Loaded", len(fvc_templates), "templates.")

N_ENROLL = min(300, len(fvc_templates))
enrolled = enroll_templates(fvc_templates[:N_ENROLL], pin="135790") 
print("Enrolled", len(enrolled), "templates.")

test_set = fvc_templates[N_ENROLL:]
y_true = []
y_scores = []
pairs_done = 0

print("Running verification trials on test templates...")
for sid, tbytes, mins, path in test_set:

    best_score = 0.0
    best_sid = None
    for rec in enrolled:
     
        s = verify_one(rec, tbytes, mins, pin="135790")
        if s > best_score:
            best_score = s
            best_sid = rec["sid"]
 
    y_true.append(1 if best_sid == sid else 0)
    y_scores.append(best_score)
    pairs_done += 1
    if pairs_done % 50 == 0:
        print("Trials done:", pairs_done)

print("Trials completed:", pairs_done)

threshold = 0.28
y_pred = [1 if s >= threshold else 0 for s in y_scores]

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix (rows=true, cols=pred):\n", cm)

plt.figure(figsize=(4,4))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xticks([0,1], ["Impostor","Genuine"])
plt.yticks([0,1], ["Impostor","Genuine"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i,j], ha="center", va="center")
plt.colorbar()
plt.show()


fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr)
plt.title(f"ROC Curve (AUC={roc_auc:.4f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()


import pickle
with open("enrolled_records.pkl", "wb") as f:
    pickle.dump(enrolled, f)
print("Enrolled records saved to enrolled_records.pkl")


if os.path.exists(SOCOFING_PATH):
  
    soco_templates = []
    soco_real_folder = os.path.join(SOCOFING_PATH, "Real")
    if os.path.exists(soco_real_folder):
        for fname in sorted(os.listdir(soco_real_folder))[:200]:  
            p = os.path.join(soco_real_folder, fname)
            tbytes, mins = process_image_to_template_bytes(p)
            soco_templates.append((p,tbytes,mins))
    print("Loaded", len(soco_templates), "SOCOFing real templates for ablation demo.")

    soco_scores = []
    for p,tbytes,mins in soco_templates:
        best_score=0.0
        for rec in enrolled:
            s = verify_one(rec, tbytes, mins, pin="135790")
            if s > best_score:
                best_score = s
        soco_scores.append(best_score)

    plt.figure()
    plt.hist(soco_scores, bins=30)
    plt.title("SOCOFing: Match Scores Distribution (Ablation Demo)")
    plt.xlabel("Match score")
    plt.ylabel("Count")
    plt.show()
else:
    print("SOCOFING dataset path not found; skip ablation demo.")

