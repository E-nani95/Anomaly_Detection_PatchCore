import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from pathlib import Path
import numpy as np
import time
import shutil
from tqdm import tqdm
import cv2
import csv  # 📌 [추가] CSV 모듈 임포트

from torch.utils.data import Dataset, DataLoader
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import ImageBatch
from torchmetrics.classification import BinaryConfusionMatrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score
import glob

# =================================================================================
# --- ### 사용자 설정 ### ---
# =================================================================================
DATASET_ROOT = Path('C:/Users/AI-00/Desktop/Dataset/Deepfake')
IMAGE_SIZE = 256
BATCH_SIZE = 32
PROJECT_ROOT = Path("anomalib_local_results_PatchCore")


# =================================================================================

def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)

    return f"{mins}분 {secs}초"


# =================================================================================
# --- ### LocalFolderDataset 클래스 (Segmentation 지원) ### ---
# =================================================================================
class LocalFolderDataset(Dataset):
    def __init__(self, root_dir, phase, transform=None):
        self.root_dir = Path(root_dir)
        self.phase = phase
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.mask_paths = []

        self._load_dataset()

    def _load_dataset(self):
        if self.phase == 'train':
            image_dir = self.root_dir / 'train' / 'good'
            paths = sorted(list(image_dir.glob('*.*')))
            self.image_paths.extend(paths)
            self.labels.extend([0] * len(paths))
            self.mask_paths.extend([None] * len(paths))

        elif self.phase == 'test':
            good_dir = self.root_dir / 'test' / 'good'
            good_paths = sorted(list(good_dir.glob('*.*')))
            self.image_paths.extend(good_paths)
            self.labels.extend([0] * len(good_paths))
            self.mask_paths.extend([None] * len(good_paths))

            bad_dir = self.root_dir / 'test' / 'bad'
            mask_dir = self.root_dir / 'test' / 'mask_landmark'
            bad_paths = sorted(list(bad_dir.glob('*.*')))
            self.image_paths.extend(bad_paths)
            self.labels.extend([1] * len(bad_paths))

            for path in bad_paths:
                mask_path = next(mask_dir.glob(f"{path.stem}.*"), None)
                self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 128).astype(np.uint8)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return {
            'image_path': str(image_path),
            'image': image,
            'label': label,
            'mask': mask.unsqueeze(0)
        }


def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    image_paths = [item['image_path'] for item in batch]
    return ImageBatch(image=images, gt_label=labels, gt_mask=masks, image_path=image_paths)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("--- 1. 데이터 준비 (로컬 폴더 로더 사용) ---")
    transform_pipeline = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    train_dataset = LocalFolderDataset(root_dir=DATASET_ROOT, phase='train', transform=transform_pipeline)
    test_dataset = LocalFolderDataset(root_dir=DATASET_ROOT, phase='test', transform=transform_pipeline)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print(f"오류: '{DATASET_ROOT}' 경로에 데이터가 없습니다. 폴더 구조를 확인하세요.")
        return

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                              collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    print("\n--- 2. 모델 및 엔진 초기화 ---")
    model = Patchcore()
    engine = Engine(default_root_dir=PROJECT_ROOT, accelerator=device, devices=1)

    print("\n--- 3. 모델 학습 ---")
    start_time_train = time.time()
    engine.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    end_time_train = time.time()
    print(f"✅ 학습 완료! (총 소요 시간: {format_time(end_time_train - start_time_train)})")

    print("\n--- 4. 성능 평가 및 시각화 생성 ---")
    start_time_test = time.time()
    engine.test(model=model, dataloaders=test_loader)
    end_time_test = time.time()
    print(f"✅ 성능 평가 완료! (총 소요 시간: {format_time(end_time_test - start_time_test)})")

    print("\n--- 5. 상세 예측 결과 수집 ---")
    start_time_predict = time.time()
    predictions = engine.predict(model=model, dataloaders=test_loader)
    end_time_predict = time.time()
    print(f"✅ 예측 완료! (총 소요 시간: {format_time(end_time_predict - start_time_predict)})")

    all_pred_labels, all_gt_labels, all_pred_scores, all_image_paths = [], [], [], []
    if not predictions or len(predictions) == 0:
        print("예측 결과가 없습니다. 모델이나 데이터 로더를 확인하세요.")
        return

    for batch in predictions:
        all_pred_labels.extend(batch.pred_label.cpu().numpy())
        all_gt_labels.extend(batch.gt_label.cpu().numpy())
        all_pred_scores.extend(batch.pred_score.cpu().numpy())
        all_image_paths.extend(batch.image_path)

    tn, fp, fn, tp = confusion_matrix(all_gt_labels, all_pred_labels, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = f1_score(all_gt_labels, all_pred_labels)
    auroc = roc_auc_score(all_gt_labels, all_pred_scores)
    print("\n" + "=" * 50)
    print("종합 성능 리포트 (이미지 레벨)")
    print("=" * 50)
    print("Confusion Matrix:")
    print(f"  - TN (정상->정상): {tn}")
    print(f"  - FP (정상->불량): {fp}  <-- 오탐지")
    print(f"  - TP (불량->불량): {tp}")
    print(f"  - FN (불량->정상): {fn}  <-- 미탐지")
    print("-" * 50)
    print("주요 성능 지표:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - F1-Score: {f1:.4f}")
    print(f"  - AUROC: {auroc:.4f}")
    print("=" * 50)
    print("ℹ️  픽셀 레벨(Segmentation) 성능은 위 '성능 평가' 단계의 로그를 확인하세요.")

    # 📌 ============================================================================
    # --- ### [추가] 오탐지(FP)/미탐지(FN) 파일 리스트 CSV로 저장 ### ---
    # ============================================================================
    print("\n--- 오탐지/미탐지 리스트 저장 중 ---")
    log_dir = Path(engine.trainer.log_dir)
    csv_path = log_dir / "misclassified_report.csv"

    fp_files = []
    fn_files = []

    for i in range(len(all_image_paths)):
        gt = all_gt_labels[i]
        pred = all_pred_labels[i]
        image_path = all_image_paths[i]

        if gt == 0 and pred == 1:
            fp_files.append(image_path)

        elif gt == 1 and pred == 0:
            fn_files.append(image_path)

    try:
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['Classification_Type', 'File_Path'])

            for path in fp_files:
                writer.writerow(['False_Positive', path])

            for path in fn_files:
                writer.writerow(['False_Negative', path])

        print(f"✅ 오탐지/미탐지 리포트 저장 완료: {csv_path.resolve()}")

    except Exception as e:
        print(f"❌ CSV 파일 저장 중 오류 발생: {e}")
    # ============================================================================

    print("\n--- 6. 결과 이미지 자동 분류 ---")
    start_time_classify = time.time()
    image_save_path = Path(engine.trainer.log_dir) / "images"

    if not image_save_path.exists():
        print(f"결과 이미지가 저장된 경로를 찾을 수 없습니다: {image_save_path}")
        return

    search_paths = [
        image_save_path / "good",
        image_save_path / "bad",
        image_save_path
    ]
    existing_search_paths = [p for p in search_paths if p.exists()]

    if not existing_search_paths:
        print(f"이미지를 찾을 수 있는 하위 폴더(good, bad)가 없습니다: {image_save_path}")
        return

    tn_dir, fp_dir, tp_dir, fn_dir = (
        image_save_path / "TN", image_save_path / "FP", image_save_path / "TP", image_save_path / "FN")
    os.makedirs(tn_dir, exist_ok=True)
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(tp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)

    for i in tqdm(range(len(all_image_paths)), desc="결과 이미지 분류 중"):
        gt = all_gt_labels[i]
        pred = all_pred_labels[i]
        target_dir = None
        if gt == 0 and pred == 0:
            target_dir = tn_dir
        elif gt == 0 and pred == 1:
            target_dir = fp_dir
        elif gt == 1 and pred == 1:
            target_dir = tp_dir
        elif gt == 1 and pred == 0:
            target_dir = fn_dir

        if target_dir:
            base_stem = Path(all_image_paths[i]).stem
            for search_dir in existing_search_paths:
                search_pattern = str(search_dir / f"*{base_stem}*.png")
                for src_path_str in glob.glob(search_pattern):
                    src_path = Path(src_path_str)
                    dst_path = target_dir / src_path.name
                    try:
                        shutil.move(str(src_path), str(dst_path))
                    except FileNotFoundError:
                        print(f"경고: 파일을 이동하는 중 찾을 수 없습니다: {src_path}")

    end_time_classify = time.time()
    print(f"✅ 자동 분류 완료! (총 소요 시간: {format_time(end_time_classify - start_time_classify)})")

    print("\n[분석 방법]")
    print("아래 경로의 하위 폴더(TN, FP, TP, FN)에서 각 케이스에 해당하는 히트맵 이미지를 확인하세요:")
    print(f"➡️  {image_save_path.resolve()}")
    print("=" * 50)


if __name__ == '__main__':
    main()