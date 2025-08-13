import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import manga_ocr
import transformers
import glob
import torch

YOLO_MODEL_PATH = 'best.pt'
INPUT_DIR = './input'
OUTPUT_DIR = './output'
FONT_PATH = 'font.ttf'
TRANSLATION_MODEL_ID = "Helsinki-NLP/opus-mt-ja-en"
# Add your Hugging Face User Access Token here if needed (usually not required for this model)
HF_TOKEN = None


def is_mostly_english(text, threshold=0.9):
    if not text.strip():
        return True 
    if len(text) == 0:
        return True
        
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    return (ascii_chars / len(text)) >= threshold

def create_mask_from_detections(image_size, detection_results):
    mask_img = Image.new('L', image_size, 0)
    mask_draw = ImageDraw.Draw(mask_img)
    if not detection_results or not hasattr(detection_results[0], 'boxes'):
        return mask_img
    for box in detection_results[0].boxes.xyxy.cpu().numpy():
        mask_draw.rectangle(box, fill="white")
    return mask_img

def extract_and_translate_text(original_image, detection_results, ocr, translator):
    translations = []
    boxes = detection_results[0].boxes.xyxy.cpu().numpy()
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        bubble_crop = original_image.crop((x1, y1, x2, y2))
        detected_text = ocr(bubble_crop)
        
        if not detected_text.strip():
            continue
        
        if is_mostly_english(detected_text):
            print(f"  > Detected (already English): '{detected_text}'")
            translated_text = detected_text
        else:
            result = translator(detected_text)
            translated_text = result[0]['translation_text']
            print(f"  > Detected: '{detected_text}' -> Translated: '{translated_text}'")
        
        translations.append({
            "box": (x1, y1, x2, y2),
            "original": detected_text,
            "translated": translated_text
        })
        
    return translations

def draw_translated_text(image, translations, font_path):
    draw = ImageDraw.Draw(image)
    if not os.path.exists(font_path):
        print(f"WARNING: Font file not found at '{font_path}'. Text will not be drawn.")
        return image

    for item in translations:
        box = item['box']
        text = item['translated']
        
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]

        font_size = min(60, int(box_height * 0.7))
        wrapped_text = text

        while font_size > 8:
            font = ImageFont.truetype(font_path, font_size)            
            lines = []
            words = text.split()
            if not words: continue

            current_line = words[0]
            for word in words[1:]:
                if font.getlength(current_line + " " + word) <= (box_width - 20):
                    current_line += " " + word
                else:
                    lines.append(current_line)
                    current_line = word
            lines.append(current_line)
            
            wrapped_text = "\n".join(lines)
            
            text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
            text_height = text_bbox[3] - text_bbox[1]
            
            if text_height <= (box_height - 15):
                break
            
            font_size -= 2
        
        font = ImageFont.truetype(font_path, font_size)
        text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        text_x = box[0] + (box_width - text_width) / 2
        text_y = box[1] + (box_height - text_height) / 2
        
        text_color = "black"
        try:
            background_crop = image.crop(box)
            avg_brightness = np.mean(np.array(background_crop.convert('L')))
            text_color = "white" if avg_brightness < 128 else "black"
        except Exception as e:
            print(f"Could not determine background color, defaulting to black. Error: {e}")

        draw.multiline_text((text_x, text_y), wrapped_text, font=font, fill=text_color, align="center")
        
    return image

def main():
    print("Loading models...")
    device = 0 if torch.cuda.is_available() else -1
    
    bubble_detector = YOLO(YOLO_MODEL_PATH)
    ocr = manga_ocr.MangaOcr()
    
    print(f"Initializing Helsinki-NLP translator on device: {'GPU' if device == 0 else 'CPU'}")
    translator = transformers.pipeline(
        "translation",
        model=TRANSLATION_MODEL_ID,
        device=device,
        token=HF_TOKEN
    )

    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Input directory not found at '{INPUT_DIR}'. Please create it and add your images.")
        return
        
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.gif"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    if not image_paths:
        print(f"No images found in '{INPUT_DIR}'.")
        return

    print(f"\nFound {len(image_paths)} images to process.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for image_path in image_paths:
        try:
            print(f"\n--- Processing: {os.path.basename(image_path)} ---")
            
            pil_img = Image.open(image_path).convert("RGB")
            
            detection_results = bubble_detector(pil_img)
            
            translated_content = extract_and_translate_text(pil_img, detection_results, ocr, translator)
            
            mask_pil = create_mask_from_detections(pil_img.size, detection_results)
            
            original_cv2_img = cv2.imread(image_path)
            mask_cv2 = np.array(mask_pil)
            blurred_mask = cv2.GaussianBlur(mask_cv2, (21, 21), 0)
            alpha_3_channel = cv2.cvtColor(blurred_mask, cv2.COLOR_GRAY2BGR)
            alpha = alpha_3_channel.astype(np.float32) / 255.0
            original_float = original_cv2_img.astype(np.float32)
            white_float = np.full_like(original_cv2_img, (255, 255, 255), dtype=np.float32)
            foreground = cv2.multiply(alpha, white_float)
            background = cv2.multiply(1.0 - alpha, original_float)
            cleaned_cv2_img = cv2.add(foreground, background).astype(np.uint8)
            
            cleaned_pil_img = Image.fromarray(cv2.cvtColor(cleaned_cv2_img, cv2.COLOR_BGR2RGB))
            
            final_image = draw_translated_text(cleaned_pil_img, translated_content, FONT_PATH)
            
            output_filename = os.path.splitext(os.path.basename(image_path))[0] + '_translated.png'
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            final_image.save(output_path)
            print(f"✅ Successfully saved translated image to: {output_path}")

        except Exception as e:
            print(f"--- ⚠️ FAILED to process {os.path.basename(image_path)} ---")
            print(f"Error: {e}")
            continue

if __name__ == '__main__':
    main()
