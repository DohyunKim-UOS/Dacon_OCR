import os
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import pytesseract
from pdf2image import convert_from_path
import subprocess
from pathlib import Path

#새롭게 설치해야 할 라이브러리
# pip install "git+https://github.com/lukas-blecher/LaTeX-OCR"
from pix2tex.cli import LatexOCR

# 모델 로딩
yolo_model = YOLO("./model/yolov11n-doclaynet.pt")
#새로운 모델
equation_model = LatexOCR()

# 클래스 매핑
LABEL_MAP = {
    'Text': 'text',
    'Title': 'title',
    'Section-header': 'subtitle',
    'Formula': 'equation',
    'Table': 'table',
    'Picture': 'image'
}

def convert_to_images(input_path, temp_dir, dpi=200):
    ext = Path(input_path).suffix.lower()
    os.makedirs(temp_dir, exist_ok=True)

    if ext == ".pdf":
        return convert_from_path(input_path, dpi=dpi, output_folder=temp_dir, fmt="png")
    elif ext == ".pptx":
        # Convert pptx to pdf first
        subprocess.run([
            "libreoffice", "--headless", "--convert-to", "pdf", "--outdir", temp_dir, input_path
        ], check=True)
        pdf_path = os.path.join(temp_dir, Path(input_path).with_suffix(".pdf").name)
        return convert_from_path(pdf_path, dpi=dpi, output_folder=temp_dir, fmt="png")
    elif ext in [".jpg", ".jpeg", ".png"]:
        return [Image.open(input_path).convert("RGB")]
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")
    
def scale_bbox_to_target(bbox, current_size, target_size):
    x1, y1, x2, y2 = bbox
    scale_x = target_size[0] / current_size[0]
    scale_y = target_size[1] / current_size[1]
    return [
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y)
    ]
    
# OCR 정확도 높이는 방향으로 수정 요함
def extract_text(image_pil, bbox): 
    x1, y1, x2, y2 = bbox
    cropped = image_pil.crop((x1, y1, x2, y2))
    return pytesseract.image_to_string(cropped, lang='kor+eng').strip()

#개선
def extract_equation_as_latex(image_pil, bbox):
    try:
        x1, y1, x2, y2 = bbox
        cropped_image = image_pil.crop((x1, y1, x2, y2))
        
        # Pix2Tex 모델을 사용하여 이미지에서 LaTeX 코드 추출
        latex_code = equation_model(cropped_image)
        
        return f'${latex_code}$'
        
    except Exception as e:
        print(f"수식 변환 실패: {e}")
        return "" # 오류 발생 시 빈 문자열 반환
    
#개선
def inference_one_image_improved(id_val, image_pil, target_size, conf_thres=0.5):
    original_size = image_pil.size
    resized_image = image_pil.resize((1024, 1024))
    temp_path = "_temp_image.png"
    resized_image.save(temp_path)

    results = yolo_model(source=temp_path, imgsz=1024, conf=conf_thres, verbose=False)[0]
    os.remove(temp_path)

    predictions = []
    
    # 1. 모델 예측 결과를 리스트에 저장
    for box, score, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        label = results.names[int(cls)]
        if label not in LABEL_MAP:
            continue
        category_type = LABEL_MAP[label]
        bbox_scaled = scale_bbox_to_target(box.tolist(), (1024, 1024), target_size)
        
        # 💡 개선된 로직: category_type에 따라 한 번만 호출
        text = ''
        if category_type in ['title', 'subtitle', 'text']:
            text = extract_text(image_pil, bbox_scaled)
        elif category_type == 'equation':
            text = extract_equation_as_latex(image_pil, bbox_scaled, equation_model)
        
        predictions.append({
            'ID': id_val,
            'category_type': category_type,
            'confidence_score': score.cpu().item(),
            'bbox': f'{bbox_scaled[0]}, {bbox_scaled[1]}, {bbox_scaled[2]}, {bbox_scaled[3]}',
            'bbox_list': bbox_scaled, 
            'text': text
        })

    # 2. 바운딩 박스 위치를 기준으로 정렬
    predictions.sort(key=lambda p: (p['bbox_list'][1], p['bbox_list'][0]))
    
    # 3. 정렬된 리스트에 'order' 값 부여
    for i, p in enumerate(predictions):
        p['order'] = i
        del p['bbox_list']
        
    return predictions

def inference(test_csv_path="./data/test.csv", output_csv_path="./output/submission.csv"):
    output_dir = os.path.dirname(output_csv_path)
    os.makedirs(output_dir, exist_ok=True)

    temp_image_dir = "./temp_images"
    os.makedirs(temp_image_dir, exist_ok=True)

    csv_dir = os.path.dirname(test_csv_path)
    test_df = pd.read_csv(test_csv_path)
    all_preds = []

    for _, row in test_df.iterrows():
        id_val = row['ID']
        raw_path = row['path']
        file_path = os.path.normpath(os.path.join(csv_dir, raw_path))
        target_width = int(row['width'])
        target_height = int(row['height'])

        if not os.path.exists(file_path):
            print(f"⚠️ 파일 없음: {file_path}")
            continue

        try:
            images = convert_to_images(file_path, temp_image_dir)
            for i, image in enumerate(images):
                full_id = f"{id_val}_p{i+1}" if len(images) > 1 else id_val
                preds = inference_one_image_improved(full_id, image, (target_width, target_height))
                all_preds.extend(preds)
            print(f"✅ 예측 완료: {file_path}")
        except Exception as e:
            print(f"❌ 처리 실패: {file_path} → {e}")

    result_df = pd.DataFrame(all_preds)
    result_df.to_csv(output_csv_path, index=False, encoding='UTF-8-sig')
    print(f"✅ 저장 완료: {output_csv_path}")
    
if __name__ == "__main__":
    inference("./data/test.csv", "./output/submission.csv")