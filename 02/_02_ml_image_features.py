# [이미지 실습] ⑤ Feature Engineering 시대 재현 — Edge · Corner · HOG · SIFT · Color · Texture
#   실습 1: Edge(Canny) · Corner(Harris)         — "윤곽과 꼭짓점을 보자"
#   실습 2: HOG · SIFT                            — "국소 기울기 방향의 분포를 보자"
#   실습 3: Color histogram · Texture(LBP, GLCM)  — "색 분포와 결(무늬)을 보자"
#   실습 4: 손글씨 digits — Raw pixel vs HOG 특징 → SVM
#   실습 5: 텍스처 3종(brick/grass/gravel) 패치 — Raw vs Color vs LBP vs GLCM → RandomForest
# 필요 패키지: scikit-image>=0.19, scikit-learn, matplotlib  (모든 이미지는 skimage 내장 → 다운로드 없음)
#
# 실행법:
#   python fe_image_features.py              # 단계마다 그림 창을 띄움(창을 닫으면 다음 단계로 진행)
#   python fe_image_features.py fe_figs      # 저장 폴더 지정
#   python fe_image_features.py fe_figs --noshow   # 창 없이 저장만 (예전 동작)
import os, sys, subprocess
import numpy as np
import matplotlib

# ---------------------------------------------------------------
# 0. 출력 설정 — 단계마다 그림을 "화면에 표시 + 파일로 저장"
# ---------------------------------------------------------------
ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
FLAGS = {a for a in sys.argv[1:] if a.startswith("-")}
OUT = ARGS[0] if ARGS else "fe_figs"
WANT_SHOW = "--noshow" not in FLAGS

def _pick_backend():
    """가능하면 창을 띄우는 백엔드, 없으면 Agg(저장 전용)로 자동 전환"""
    if not WANT_SHOW:
        matplotlib.use("Agg"); return "Agg", False
    if "ipykernel" in sys.modules:                 # Jupyter → inline 으로 그대로 표시
        return matplotlib.get_backend(), True
    for bk in ("TkAgg", "QtAgg", "Qt5Agg", "MacOSX"):
        try:
            matplotlib.use(bk); return bk, True
        except Exception:
            continue
    matplotlib.use("Agg"); return "Agg", False     # GUI 없음 → 저장 후 기본 뷰어로 열기

BACKEND, CAN_SHOW = _pick_backend()
import matplotlib.pyplot as plt

def _open_file(path):
    """GUI 백엔드가 없을 때 저장된 PNG를 OS 기본 이미지 뷰어로 열어 준다"""
    try:
        if sys.platform.startswith("win"):   os.startfile(os.path.abspath(path))
        elif sys.platform == "darwin":       subprocess.run(["open", path], check=False)
        else:                                subprocess.run(["xdg-open", path], check=False)
    except Exception:
        pass

def show(fig, fname):
    """그림을 저장하고 곧바로 화면에 표시 (창을 닫으면 다음 단계로 진행)"""
    path = os.path.join(OUT, fname)
    fig.savefig(path, dpi=110)
    print(f"   -> 그림: {path}" + ("  (창을 닫으면 다음 단계로 진행)" if CAN_SHOW else ""))
    if CAN_SHOW:
        plt.show()
    elif WANT_SHOW:
        _open_file(path)
    plt.close(fig)

from skimage import data, color, exposure, transform, util
from skimage.feature import (canny, corner_harris, corner_peaks, hog, SIFT,
                             local_binary_pattern, graycomatrix, graycoprops)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_digits

rng = np.random.default_rng(0)
os.makedirs(OUT, exist_ok=True)
from matplotlib import font_manager as _fm
_avail = {f.name for f in _fm.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in ["Malgun Gothic", "Noto Sans CJK JP", "Noto Sans CJK KR", "AppleGothic"] if f in _avail), "DejaVu Sans")
plt.rcParams["axes.unicode_minus"] = False
print(f"[표시 설정] matplotlib 백엔드 = {BACKEND}, 화면 표시 = {'예' if CAN_SHOW else '아니오(파일 저장' + (' 후 뷰어로 열기)' if WANT_SHOW else '만)')}")

# 공통 이미지: 우주비행사(컬러) · 흑백 변환
img_rgb = data.astronaut()                     # (512, 512, 3)
img = color.rgb2gray(img_rgb)                  # 0~1 float

# ---- 그림 0: 입력 원본 ----
fig, ax = plt.subplots(1, 2, figsize=(8, 4.2))
ax[0].imshow(img_rgb); ax[0].set_title(f"원본 컬러 {img_rgb.shape}")
ax[1].imshow(img, cmap="gray"); ax[1].set_title(f"흑백 변환 {img.shape}")
for a in ax: a.axis("off")
fig.suptitle("[입력] skimage astronaut — 이 이미지에서 특징을 뽑는다")
fig.tight_layout(); show(fig, "fig0_input.png")

# ---------------------------------------------------------------
# 실습 1. Edge (Canny 1986) · Corner (Harris 1988)
# ---------------------------------------------------------------
edges = canny(img, sigma=2.0)                                   # True/False 경계 맵
harris = corner_harris(img)                                     # 코너 응답 맵
corners = corner_peaks(harris, min_distance=10, threshold_rel=0.02)
print(f"[실습 1] Edge 픽셀 수 = {edges.sum():,} / {edges.size:,}   Corner 개수 = {len(corners)}")

# ---- 그림 1: Edge · Corner ----
fig, ax = plt.subplots(1, 4, figsize=(15, 4.2))
ax[0].imshow(img, cmap="gray"); ax[0].set_title("입력 (흑백)")
ax[1].imshow(edges, cmap="gray"); ax[1].set_title(f"Edge — Canny σ=2.0\n({edges.sum():,} px, 전체의 {edges.mean()*100:.1f}%)")
ax[2].imshow(exposure.rescale_intensity(harris, out_range=(0, 1)), cmap="magma"); ax[2].set_title("Harris 코너 응답 맵")
ax[3].imshow(img, cmap="gray"); ax[3].plot(corners[:, 1], corners[:, 0], "r.", ms=6); ax[3].set_title(f"Corner — Harris ({len(corners)}개)")
for a in ax: a.axis("off")
fig.suptitle("[실습 1] 윤곽(Edge)과 꼭짓점(Corner)")
fig.tight_layout(); show(fig, "fig1_edge_corner.png")

# ---------------------------------------------------------------
# 실습 2. HOG (Dalal & Triggs 2005) · SIFT (Lowe 2004)
# ---------------------------------------------------------------
small = transform.resize(img, (128, 128))
hog_vec, hog_img = hog(small, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=True)
print(f"[실습 2] HOG 특징 벡터 길이 = {len(hog_vec)}  (128×128 이미지 → 숫자 {len(hog_vec)}개)")
sift = SIFT()
sift.detect_and_extract(util.img_as_ubyte(img))
print(f"[실습 2] SIFT 키포인트 = {len(sift.keypoints)}개, 각 키포인트 기술자 = {sift.descriptors.shape[1]}차원")

# ---- 그림 2: HOG · SIFT ----
fig, ax = plt.subplots(1, 4, figsize=(15, 4.2))
ax[0].imshow(small, cmap="gray"); ax[0].set_title("입력 128×128"); ax[0].axis("off")
ax[1].imshow(exposure.rescale_intensity(hog_img, in_range=(0, 10)), cmap="gray")
ax[1].set_title(f"HOG 시각화 ({len(hog_vec)}차원)"); ax[1].axis("off")
ax[2].plot(hog_vec[:200], lw=0.9, color="#1E1B4B")
ax[2].set_title("HOG 벡터 앞 200개 값"); ax[2].set_xlabel("특징 index"); ax[2].set_ylabel("크기")
kp = sift.keypoints[:300]
ax[3].imshow(img, cmap="gray"); ax[3].scatter(kp[:, 1], kp[:, 0], s=10, c="cyan", alpha=0.8)
ax[3].set_title(f"SIFT 키포인트 ({len(sift.keypoints)}개 중 300)\n기술자 {sift.descriptors.shape[1]}차원"); ax[3].axis("off")
fig.suptitle("[실습 2] 국소 기울기 방향의 분포 — HOG · SIFT")
fig.tight_layout(); show(fig, "fig2_hog_sift.png")

# ---------------------------------------------------------------
# 실습 3. Color histogram · Texture (LBP: Ojala 2002, GLCM: Haralick 1973)
# ---------------------------------------------------------------
def color_hist(rgb, bins=8):
    """채널별 히스토그램을 이어붙인 3×bins 차원 색 특징"""
    return np.concatenate([np.histogram(rgb[..., c], bins=bins, range=(0, 255))[0] for c in range(3)]).astype(float)

def lbp_hist(gray_u8, P=8, R=1):
    """Local Binary Pattern 히스토그램 (P+2 차원, uniform)"""
    lbp = local_binary_pattern(gray_u8, P, R, method="uniform")
    h, _ = np.histogram(lbp, bins=P + 2, range=(0, P + 2))
    return h / h.sum()

def glcm_feats(gray_u8):
    """GLCM 통계 4종: contrast, homogeneity, energy, correlation"""
    g = graycomatrix(gray_u8, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                     levels=256, symmetric=True, normed=True)
    return np.array([graycoprops(g, p).mean() for p in ["contrast", "homogeneity", "energy", "correlation"]])

ch = color_hist(img_rgb)
img_u8 = util.img_as_ubyte(img)
lbp_img = local_binary_pattern(img_u8, 8, 1, method="uniform")
lbp_h = lbp_hist(img_u8)
glcm_v = glcm_feats(img_u8)
print(f"[실습 3] Color 특징 = {len(ch)}차원, LBP 히스토그램 = {len(lbp_h)}차원, GLCM = 4차원")

# ---- 그림 3: 색 분포 · 결(무늬) ----
fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
ax[0, 0].imshow(img_rgb); ax[0, 0].set_title("원본 (색 정보)"); ax[0, 0].axis("off")
for c, name in enumerate(["R", "G", "B"]):
    ax[0, 1].plot(ch[c * 8:(c + 1) * 8], marker="o", label=name,
                  color={"R": "#F43F5E", "G": "#10B981", "B": "#3B82F6"}[name])
ax[0, 1].set_title("Color histogram (8 bins × 3채널 = 24차원)")
ax[0, 1].set_xlabel("밝기 구간"); ax[0, 1].set_ylabel("픽셀 수"); ax[0, 1].legend()
ax[0, 2].bar(range(len(ch)), ch, color=["#F43F5E"] * 8 + ["#10B981"] * 8 + ["#3B82F6"] * 8)
ax[0, 2].set_title("이어붙인 24차원 색 특징 벡터"); ax[0, 2].set_xlabel("특징 index")
ax[1, 0].imshow(lbp_img, cmap="gray"); ax[1, 0].set_title("Texture — LBP 맵 (uniform, P=8)"); ax[1, 0].axis("off")
ax[1, 1].bar(range(len(lbp_h)), lbp_h, color="#1E1B4B")
ax[1, 1].set_title(f"LBP histogram ({len(lbp_h)}차원)"); ax[1, 1].set_xlabel("LBP 패턴 코드")
gnames = ["contrast", "homogeneity", "energy", "correlation"]
b = ax[1, 2].bar(gnames, glcm_v, color="#06B6D4")
for r, v in zip(b, glcm_v): ax[1, 2].text(r.get_x() + r.get_width() / 2, v * 1.15, f"{v:.3g}", ha="center", fontsize=9)
ax[1, 2].set_yscale("log")
ax[1, 2].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))   # 한글 폰트에 없는 유니코드 −(U+2212) 회피
ax[1, 2].set_title("GLCM 통계 4종 (log 스케일)"); ax[1, 2].tick_params(axis="x", labelsize=8)
fig.suptitle("[실습 3] 색 분포(Color histogram)와 결(LBP · GLCM)")
fig.tight_layout(); show(fig, "fig3_color_texture.png")

# ---- 그림 3b: 특징 6종 한눈에 (요약) ----
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax[0, 0].imshow(img_rgb); ax[0, 0].set_title("원본 (Raw Image)")
ax[0, 1].imshow(edges, cmap="gray"); ax[0, 1].set_title(f"Edge — Canny ({edges.sum():,} px)")
ax[0, 2].imshow(img, cmap="gray"); ax[0, 2].plot(corners[:, 1], corners[:, 0], "r.", ms=5); ax[0, 2].set_title(f"Corner — Harris ({len(corners)}개)")
ax[1, 0].imshow(exposure.rescale_intensity(hog_img, in_range=(0, 10)), cmap="gray"); ax[1, 0].set_title(f"HOG ({len(hog_vec)}차원)")
ax[1, 1].imshow(img, cmap="gray"); ax[1, 1].scatter(kp[:, 1], kp[:, 0], s=8, c="cyan"); ax[1, 1].set_title(f"SIFT 키포인트 ({len(sift.keypoints)}개 중 300)")
ax[1, 2].imshow(lbp_img, cmap="gray"); ax[1, 2].set_title("Texture — LBP (uniform, P=8)")
for a in ax.ravel(): a.axis("off")
fig.suptitle("[요약] 손으로 설계한 특징 6종")
fig.tight_layout(); show(fig, "fig3b_features6.png")

# ---------------------------------------------------------------
# 실습 4. 손글씨 digits: Raw pixel(64) vs HOG 특징(32) → SVM
#   + 테스트 이미지를 ±1픽셀 이동시켜 "위치가 조금 달라진" 상황도 평가 (특징의 불변성 확인)
# ---------------------------------------------------------------
digits = load_digits()
yd = digits.target
def shift(im, dx, dy):                       # 이미지를 (dx, dy) 픽셀 만큼 이동
    return np.roll(np.roll(im, dy, 0), dx, 1)
def hog16(im):                               # 8×8 → 16×16 확대 후 8×8 셀 HOG (32차원)
    return hog(transform.resize(im, (16, 16), order=1), orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
def feats(ims):
    return ims.reshape(len(ims), -1), np.array([hog16(im) for im in ims])

Itr, Ite, ytr, yte = train_test_split(digits.images, yd, test_size=0.3, random_state=42, stratify=yd)
Ite_shift = np.array([shift(im, *rng.integers(-1, 2, 2)) for im in Ite])   # 테스트만 ±1px 이동
(raw_tr, hog_tr), (raw_te, hog_te), (raw_sh, hog_sh) = feats(Itr), feats(Ite), feats(Ite_shift)
res4 = {}
for name, (a, b, c) in {"Raw pixel (64)": (raw_tr, raw_te, raw_sh), "HOG (32)": (hog_tr, hog_te, hog_sh)}.items():
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5)).fit(a, ytr)
    res4[name] = clf.score(b, yte); res4[name + " ±1px 이동"] = clf.score(c, yte)
print("[실습 4] digits SVM 정확도:", {k: round(v, 3) for k, v in res4.items()})

# ---- 그림 4: digits 원본 · 이동 · HOG + 정확도 ----
fig = plt.figure(figsize=(13, 6.2))
gs = fig.add_gridspec(3, 8, width_ratios=[1] * 6 + [0.35, 2.2])
for i in range(6):
    a0 = fig.add_subplot(gs[0, i]); a0.imshow(digits.images[i], cmap="gray"); a0.axis("off")
    a1 = fig.add_subplot(gs[1, i]); a1.imshow(shift(digits.images[i], 1, 0), cmap="gray"); a1.axis("off")
    _, hv = hog(transform.resize(digits.images[i], (16, 16), order=1), orientations=8,
                pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True)
    a2 = fig.add_subplot(gs[2, i]); a2.imshow(hv, cmap="gray"); a2.axis("off")
    if i == 0:
        a0.set_title("원본 8×8", fontsize=9); a1.set_title("→1px 이동", fontsize=9); a2.set_title("HOG 시각화", fontsize=9)
    else:
        a0.set_title(f"y={digits.target[i]}", fontsize=9)
ab = fig.add_subplot(gs[:, 7])
bars = ab.bar(list(res4.keys()), list(res4.values()), color=["#9CA3AF", "#D1D5DB", "#1E1B4B", "#06B6D4"])
for r, v in zip(bars, res4.values()): ab.text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
ab.set_ylim(0, 1.12); ab.set_title("SVM 정확도"); ab.tick_params(axis="x", labelsize=7, rotation=30)
fig.suptitle("[실습 4] digits — 이동해도 HOG 패턴은 비슷하다 (Raw는 크게 무너진다)")
fig.tight_layout(); show(fig, "fig4_digits_hog.png")

# ---------------------------------------------------------------
# 실습 5. 텍스처 3종 패치 분류: Raw vs Color vs LBP vs GLCM
# ---------------------------------------------------------------
tex = {"brick": data.brick(), "grass": data.grass(), "gravel": data.gravel()}   # 512×512 gray
P = 64
patches, labels = [], []
for li, (name, im) in enumerate(tex.items()):
    for _ in range(80):
        r, c = rng.integers(0, 512 - P, 2)
        patches.append(im[r:r + P, c:c + P]); labels.append(li)
patches, labels = np.array(patches), np.array(labels)
feat_sets = {
    "Raw pixel (4096)": patches.reshape(len(patches), -1).astype(float),
    "Gray hist (16)":   np.array([np.histogram(p, bins=16, range=(0, 255))[0] for p in patches], float),
    "LBP hist (10)":    np.array([lbp_hist(p) for p in patches]),
    "GLCM (4)":         np.array([glcm_feats(p) for p in patches]),
}
res5 = {}
for name, Xt in feat_sets.items():
    Xtr, Xte, ytr, yte = train_test_split(Xt, labels, test_size=0.3, random_state=42, stratify=labels)
    clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xtr, ytr)
    res5[name] = clf.score(Xte, yte)
print("[실습 5] 텍스처 3종 RF 정확도:", {k: round(v, 3) for k, v in res5.items()})

# ---- 그림 5: 텍스처 원본 · 학습에 쓴 64×64 패치 · LBP 히스토그램 ----
fig, ax = plt.subplots(3, 3, figsize=(10, 9))
for i, (name, im) in enumerate(tex.items()):
    ax[0, i].imshow(im, cmap="gray"); ax[0, i].set_title(f"{name} 원본 512×512"); ax[0, i].axis("off")
    sample = np.hstack([patches[labels == i][j] for j in range(3)])   # 64×64 패치 3장
    ax[1, i].imshow(sample, cmap="gray"); ax[1, i].set_title(f"{name} 64×64 패치 예시 (80장 중 3장)", fontsize=9); ax[1, i].axis("off")
    ax[2, i].bar(range(10), lbp_hist(im), color="#1E1B4B"); ax[2, i].set_title(f"LBP histogram — {name}", fontsize=9); ax[2, i].set_ylim(0, 0.45)
fig.suptitle("[실습 5] 텍스처 3종 — 원본 · 학습 패치 · LBP 히스토그램(결의 지문)")
fig.tight_layout(); show(fig, "fig5_texture_lbp.png")

# ---- 그림 6: 정확도 비교(실습 4 · 5) ----
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for a, res, title in [(ax[0], res4, "실습 4 digits (SVM)"), (ax[1], res5, "실습 5 텍스처 3종 (RandomForest)")]:
    cols = (["#9CA3AF", "#D1D5DB", "#1E1B4B", "#06B6D4"] if len(res) == 4 else ["#9CA3AF"] + ["#1E1B4B"] * (len(res) - 1))
    bars = a.bar(list(res.keys()), list(res.values()), color=cols)
    for b, v in zip(bars, res.values()): a.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    a.set_ylim(0, 1.12); a.set_title(title); a.tick_params(axis="x", labelsize=7.5)
fig.suptitle("[비교] 어떤 특징이 더 잘 통했나")
fig.tight_layout(); show(fig, "fig6_accuracy.png")

print("그림 저장 폴더:", os.path.abspath(OUT))
for f in sorted(os.listdir(OUT)):
    print("  -", f)
