from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pandas as pd
from PIL import Image
import os, requests

url = "https://pokemonkorea.co.kr/pokedex"
class_list = ["노말", "불꽃", "물", "풀", "전기", "얼음", "격투", "독", "땅", "비행", "에스퍼", "벌레", "바위", "고스트", "드래곤", "악", "강철", "페어리"]

df = pd.DataFrame(columns=['img_path'] + class_list)

browser = webdriver.Chrome()
browser.get(url)

prev_height = browser.execute_script("return document.body.scrollHeight")
while True:
	browser.execute_script("window.scrollTo(0,document.body.scrollHeight)")
	time.sleep(0.5)
	current_height = browser.execute_script("return document.body.scrollHeight")
	if current_height == prev_height: break
	prev_height = current_height

items = browser.find_elements(By.CLASS_NAME, "col-lg-2.col-6")
print(f"count : {len(items)}")

os.makedirs('dataset/images', exist_ok=True)
for i, item in enumerate(items) :
	if i % (len(items) // 10) == 0 : print(f"{i} / {len(items)}")

	description = item.find_elements(By.TAG_NAME, 'p')[-1].text
	if '거다이맥스' in description : continue

	img_src = item.find_element(By.TAG_NAME, 'img').get_attribute('src')
	img_src = img_src.replace('mid', 'full')

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

	base = os.path.basename(img_src)
	fn, ext = os.path.splitext(base)
	bg.save(f"dataset/images/{fn}.png")

	feature_vector = [0] * len(class_list)
	features = item.find_elements(By.TAG_NAME, 'span')
	for feature in features :
		feature_name = feature.text
		idx = class_list.index(feature_name)
		feature_vector[idx] = 1

	item_info = [f"{fn}.png"] + feature_vector
	item_info

	df.loc[len(df)] = item_info

df.to_csv('dataset/info.csv')
print(f"total : {len(df)}")


# img = Image.open(requests.get('https://data1.pokemonkorea.co.kr/newdata/pokedex/full/000201.png', stream=True, timeout=5).raw)
# img.convert('RGB')
# img.convert('RGB').save('problem.png')