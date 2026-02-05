import kagglehub
import csv
import re
import pandas as pd
from bs4 import BeautifulSoup
import requests
import time
from playwright.sync_api import sync_playwright
import os

def createSet(path2,path, username,password, title):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    with requests.Session() as s:
        r = s.get(path2, headers=headers, allow_redirects=True, timeout=30)
    
        if r.status_code != 200:
            #print(r.text[:500])
            return

        soup = BeautifulSoup(r.text, "html.parser")
        article = soup.find("article")
        if not article:
            
            return

        jobs = [li.get_text(strip=True) for li in article.find_all("li")]
        
    for job in jobs:
        getQualifications(path, username,password, title)


    
    

def getQualifications(path,username, password, title):
    
    with sync_playwright() as p:
        browser = p.firefox.launch(
            headless=True
        )
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080}
        )                                 
        page = context.new_page()
        page.goto(
            path,
            wait_until="domcontentloaded",
            timeout=60_000
        )
        
        page.locator('input#username').fill(username)
        page.locator('input#password').fill(password)
        page.locator('button[type="submit"]').click()

        search = page.locator('input[placeholder="Search"]')
        
        #search.fill(title)
        
        search.fill("Software Engineer")
        search.press("Enter")
        #page.locator('button[type="Jobs"]').click()
        page.get_by_role("button", name="Jobs").click()
        jobEntries = page.locator('li[role="listitem"]')

        page.wait_for_selector('a[href*="/jobs/view/"]', timeout=60_000)
        jobLink = page.locator('a[href*="/jobs/view/"]')
        count = jobLink.count()
        seen = set()
        
        for i in range(count):
            #page.locator('li:has(strong:has-text("Software Engineer"))').first.click()
            link = jobLink.nth(i)
            href = link.get_attribute("href")
            
            if href in seen:
                continue
            seen.add(href)
            link.click()
            page.wait_for_timeout(6000)
            #use beautifulSoup here to physically scrape the data
            #then the algorithms will used in the proposal
            print("good")
            page.go_back(wait_until="domcontentloaded")
        
        
        
        page.pause()
        context.close()
        browser.close()
    
    
if __name__  == "__main__":
    path = 'https://www.linkedin.com/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2F'
    path2 = 'https://www.worksourceatlanta.org/job-trends/high-demand-occupations/'

    title = None
    username = 'datasetprepper@gmail.com'
    password = 'one2three!'
    createSet(path,username, password, title,path2)
    getQualifications(path,username, password, title)
    
    #path2 = kagglehub.dataset_download("ravindrasinghrana/job-description-dataset")
    #<input id="username" name="session_key"
    #<input id="password"
    #<button class="btn__primary--large from__button--floating"
