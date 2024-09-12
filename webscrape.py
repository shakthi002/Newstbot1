## This is an Autoscaper which would keep on extracting news on a given page on moneycontrol.com news website
# given a url or a list of urls untill the work is finished

# Import required libraries
from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
from datetime import timedelta
from requests.adapters import HTTPAdapter
import re
from vect_embed import *


def scrape(file_name):
    s = requests.Session()
    s.mount('https://', HTTPAdapter(max_retries=2))
    file = open(file_name,"w", encoding='utf-8')

    # Load news-section urls in a list from text document that contains the list of Moneycontrol section urls
    # you want to scrape
    urls = [line.rstrip('\n') for line in open('files\moneycontrol_urls.txt')]

    headers={
    'Referer': 'https://www.moneycontrol.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'
    }

    # Empty dictionary to store scraped information on news articles
    news_articles = {}
    news_count = 0

    # Just a counter to update the file name before saving to the disk
    times_saved = 0

    # Start the for loop over the list of section urls to scrape all the historical news from those
    # sections one-by-one
    for url in urls:

        # Page number initialized by 0 before entering the while loop for scraping articles from a single section
        p = 0

        # Start while loop to extract text features from the historical articles
        while True:

            ## One of the key challenges faced during execution. The program gets stuck after loading 10-15k articles
            # A while loop using try and except was created, which would keep on retrying on exceptions cased by any of the errors such as timeout/internet disconnected/broken links etc.
            while True:
                response = None
                print(url)
                try:
                    response = s.get(url, headers=headers, timeout=20)
                    #print(response.text)
                    break
                except requests.exceptions.RequestException as err:
                    print(f'Caught {err}... Sleeping for 80 sec and then retrying...')
                    time.sleep(80)
                    continue

            # Parse the source page to extract html tags and content using Beautiful Soup
            data = response.text
            soup = BeautifulSoup(data,'html.parser')
            articles = soup.find_all('li',{'class':'clearfix'})

            # Run for loop on all of the articles found on a given page number of a section to extract text features
            # Features extracted: Title, Link, Date, full news article text
            count=0
            for article in articles:
                if count==10:
                    break
                count+=1
                try:
                    title = article.find('h2').text
                    #print('Title: ',title)
                    link = article.find('a').get('href')
                    #print('Link: ',link)
                    date = article.find('span').text
                    #print('Published: ',date)

                except AttributeError:
                    title = 'N/A'
                    link = 'N/A'
                    date = 'N/A'

                # Used as a count checker and input to the autosave section
                if news_count == 1:
                    start_time = time.monotonic()

                # Extract full news text by making another server request using the link of the article extracted earlier
                try:
                    news_response = s.get(link, headers=headers, timeout=15)
                    # print(news_response)
                    news_data = news_response.text
                    news_soup = BeautifulSoup(news_data,'html.parser')
                    ## The problem with this website is that when the news link is parsed,
                    # it also incldues snippets of backend codes which are totally unnecessary
                    # So use If statement to extract the whole div tag first
                    if news_soup.find('div',{'class':'arti-flow'}):
                        news_text = news_soup.find('div',{'class':'arti-flow'})

                        # Then decompose the unncessarily repeating 'script' and 'style' tags from the scraped content
                        for x in news_text.find_all("script"):
                            x.decompose()
                        for y in news_text.find_all('style'):
                            y.decompose()

                        # Finally, decompose the extra standard one-liner text from the bottom of the news article and extract the clean text of the news article
                        try:
                            news_text.find_all('a')[-1].decompose()
                            news = news_text.text
                        except IndexError:
                            news = news_text.text
                    else:
                        news = 'N/A'

                # A countermeasure in place, in case there are any of the errors such as timeout/internet disconnected/broken links etc.
                except requests.exceptions.RequestException as error:
                    news = 'N/A'
                    print(f'Caught {error}... Slpeeing for 80 sec')
                    time.sleep(80)

                # Increase the count by 1 for every news scraped and appending it to the empty dictionary
                news_count+=1
                news_articles[news_count] = [title,date,news,link]
                news=news.strip()
                index_of_news = news.find('Related stories')


    # Keep everything up to the first occurrence of 'news'
                news = news[:index_of_news]
                #cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '',news)

        # Step 4: Lowercasing and whitespace removal
                cleaned_text = ' '.join(news.split())
                if news and news !='N/A' and news!='N/':
                    st=f"The news published on the date {date} with headline {title} is: \n {cleaned_text}"
                else:
                    st=f"The news published on the date {date} with headline {title}"
                
                file.write(st)
                file.write('\n \n')
                print('*')

                # A counter, that prints number of articles scraped in a multiple of 1000
                if news_count % 1000 == 0:
                    print('No. ',news_count)

            # Collect the next page url from the bottom of the page
            url_tag = soup.find('a',{'class':'last'})

            # Max number of pages to scrape from a single section on the website restricted to 15,655
            max_pages = 15655

            # While loop would break if no url is found, as in reached the end of the section itself
            try:
                if "void" in url_tag.get('href'):
                    break

                elif url_tag.get('href') and p < max_pages:
                    url = 'https://www.moneycontrol.com'+url_tag.get('href')
                    print('\n',url,'\n')
                    p+=1
                else:
                    break
                    print('\n\nNext page does not exist\n\n')
            except AttributeError:
                print('\n\nNext page does not exist\n\n')
    file.close()
    db=vector_embedding(file_name)
    