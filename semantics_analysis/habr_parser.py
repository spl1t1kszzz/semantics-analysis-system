from typing import List

import requests
from bs4 import BeautifulSoup


class Doc:
    title: str
    paragraphs: List[str]

    def __init__(self, title: str, parahraphs: List[str]):
        self.title = title
        self.paragraphs = parahraphs

    @classmethod
    def from_article(cls, article_id):
        url = f'https://habr.com/ru/articles/{article_id}'

        response = requests.get(url)

        if response.status_code != 200:
            raise ValueError(response.text)

        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.find('title').text

        divs = soup.find_all('div', {'class': 'tm-article-body'})

        paragraphs = []

        for div in divs:
            paragraphs.extend(p.text for p in div.find_all('p'))

        return Doc(title, paragraphs)


def main():
    doc = Doc.from_article(775842)

    for p in doc.paragraphs:
        print(p)


if __name__ == '__main__':
    main()
