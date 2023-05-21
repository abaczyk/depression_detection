import urllib.request
from bs4 import BeautifulSoup
import requests
import urllib.parse as urlparse
import os


def download_files(url, destination):
    response = urllib.request.urlopen(url)
    soup = BeautifulSoup(response, 'html.parser')

    for link in soup.find_all('a'):
        print(link)
        href = link.get('href')
        if href and not href.startswith('#'):  # Skip anchor links
            if len(href) == 9:
                print(f"{href[:-4]}")
                urllib.request.urlretrieve(
                    f"https://dcapswoz.ict.usc.edu/wwwdaicwoz/{href}", destination + '\\' + href)
                print(f"Downloaded: {href}")


url = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/?C=D;O=A"
if not os.path.exists('daic_woz'):
    os.mkdir('daic_woz')
    print('Made path: daic_woz')
destination_folder = os.getcwd() + '/daic_woz'
download_files(url, destination_folder)
