import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time
import pandas as pd
from PIL import Image
import os, requests
from tqdm import tqdm

URL = "https://www.pokemon.com/us/pokedex"
SAVE_DIR = 'dataset'
SIZE = 1024

browser = uc.Chrome()
browser.get(URL)
browser.implicitly_wait(10)
print('Open browser !')

while True :
    try :
        load_btn = browser.find_element(By.ID, "loadMore")
        load_btn.click()
        time.sleep(1)
        items = browser.find_element(By.CLASS_NAME, 'results').find_elements(By.CLASS_NAME, "animating")
    except :
        break
print('Total items :', len(items))

class_kor = ["노말", "불꽃", "물", "풀", "전기", "얼음", "격투", "독", "땅", "비행", "에스퍼", "벌레", "바위", "고스트", "드래곤", "악", "강철", "페어리"]
class_en = ["Normal", "Fire", "Water", "Grass", "Electric", "Ice", "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"]
df = pd.DataFrame(columns=['img_path'] + class_en)
os.makedirs(f'{SAVE_DIR}/images', exist_ok=True)
for item in tqdm(items) :
    num = item.find_element(By.CLASS_NAME, 'id').text.replace('#', '')
    abilities = [a.text for a in item.find_elements(By.CLASS_NAME, 'abilities')]
    img_src = item.find_element(By.TAG_NAME, 'img').get_attribute('src')
    img_src = img_src.replace('detail', 'full')

    while True :
        try :
            img = Image.open(requests.get(img_src, stream=True, timeout=5).raw)
            break
        except :
            print(img_src)
            time.sleep(1)

    alpha = img.split()[-1]
    bg = Image.new('RGB', img.size, color=(255,255,255))
    bg.paste(img, mask=alpha)
    bg = bg.resize((SIZE,SIZE), Image.LANCZOS)
    bg.save(f'{SAVE_DIR}/images_{SIZE}/{num}.jpg')

    feature_vector = [0] * len(class_en)
    for ability in abilities :
        idx = class_en.index(ability)
        feature_vector[idx] = 1

    item_info = [f"{num}.jpg"] + feature_vector
    df.loc[len(df)] = item_info

df.to_csv(f'{SAVE_DIR}/info.csv')
print(f"CSV saved...! row : {len(df)}")