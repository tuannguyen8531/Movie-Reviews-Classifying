import time
import chromedriver_autoinstaller
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

chromedriver_autoinstaller.install()

def read_urls_by_index(file_path, index):
    urls = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        current_index = 0
        for line in lines:
            line = line.strip()
            if line.isdigit():
                current_index = int(line)
            elif current_index == index:
                urls.append(line)
            elif current_index > index:
                break
    return urls

def add_rating_filler(urls):
    urls_with_ratings = []
    for url in urls:
        for i in range(1, 10):
            temp = url + str(i)
            urls_with_ratings.append(temp)
    return urls_with_ratings

def crawl_data(urls):
    driver = webdriver.Chrome()
    all_texts = []
    all_ratings = []

    for url in urls:
        driver.get(url)

        while True:
            try:
                load_more_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "ipl-load-more__button"))
                )
                load_more_button.click()
                WebDriverWait(driver, 5).until(
                    EC.staleness_of(load_more_button)
                )
            except:
                break
        
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        reviews = soup.find_all("div", class_="lister-item")
        for review in reviews:
            review_text = review.find('div', class_='content').find('div', class_='show-more__control').text.strip()
            rating_element = review.find('span', class_='rating-other-user-rating')
            if rating_element:
                rating = rating_element.find("span").text.strip()
            else:
                rating = "No rating"
            all_texts.append(review_text)
            all_ratings.append(rating)
            time.sleep(0.5)
    return all_ratings, all_texts

def to_tsv(index, all_ratings, all_texts):
    tsv_file = "./train" + str(index) + ".tsv"

    all_texts_processed = [text.replace("\n", " ") for text in all_texts]

    with open(tsv_file, "w", encoding="utf-8") as file:
        file.write("sentiment\treview\n")
        for rating, text in zip(all_ratings, all_texts_processed):
            file.write(f"{int(rating)}\t{str(text)}\n")

def main():
    index = int(input("Enter index: "))
    movie_urls = read_urls_by_index("./urls.txt", index)
    urls = add_rating_filler(movie_urls)
    all_ratings, all_texts = crawl_data(urls)
    to_tsv(index, all_ratings, all_texts)
    print("Done!")

if __name__ == "__main__":
    main()