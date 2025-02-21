import os
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import git
import zipfile
import io

import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin

def scrape_notes(lecture_id):
    response = requests.get(f'https://cs50.harvard.edu/ai/2023/notes/{lecture_id}/')
    response.raise_for_status()
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')

    extracted_text = ""
    
    # Keep a counter for image files to differentiate them
    image_counter = 1
    
    for element in soup.find_all(['p', 'code', 'table', 'img']):  # Add 'img' to the list of elements to find
        if element.name == 'p':
            extracted_text += element.get_text() + "\n\n"
            
        elif element.name == 'code':
            extracted_text += element.get_text() + "\n\n"
            
        elif element.name == 'table':
            for table in soup.find_all('table'):
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    row_text = ' | '.join(cell.get_text() for cell in cells)
                    extracted_text += row_text + "\n"
                extracted_text += "\n"
        
        elif element.name == 'img':  # Handle images
            img_url = element.get('src')
            img_url = urljoin(f'https://cs50.harvard.edu/ai/2023/notes/{lecture_id}/', img_url)
            img_name = os.path.basename(img_url)
            
            # Add a placeholder in the extracted text for the image
            extracted_text += f'[Image: {img_name}] located at ./scraped_data/notes/lecture_{lecture_id}/images/{img_name}\n\n'
            
            try:
                img_response = requests.get(img_url)
                img_response.raise_for_status()
                
                # Check if image directory exists, if not create one
                image_directory = f'./scraped_data/notes/lecture_{lecture_id}/images'
                if not os.path.exists(image_directory):
                    os.makedirs(image_directory)
                
                # Save the image
                with open(os.path.join(image_directory, img_name), 'wb') as img_file:
                    img_file.write(img_response.content)
                
                # Increment the image counter
                image_counter += 1

            except requests.HTTPError as http_err:
                break
                # print(f"HTTP error occurred: {http_err}")
            except Exception as err:
                print(f"An error occurred: {err}")
    
    # Write the extracted text to a file
    if not os.path.exists(f'./scraped_data/notes/lecture_{lecture_id}/'):
        os.makedirs(f'./scraped_data/notes/lecture_{lecture_id}/')
    
    with open(f'./scraped_data/notes/lecture_{lecture_id}/notes.txt', 'w', encoding='utf-8') as file:
        file.write(extracted_text)
    
    time.sleep(0.5)
    

def scrape_transcripts(lecture_id):
    response = requests.get(f'https://cdn.cs50.net/ai/2020/spring/lectures/{lecture_id}/lang/en/lecture{lecture_id}.txt')

    if not os.path.exists(f'./scraped_data/transcript/lecture_{lecture_id}'):
        os.makedirs(f'./scraped_data/transcript/lecture_{lecture_id}')

    with open(f'./scraped_data/transcript/lecture_{lecture_id}/transcript.txt', 'w') as file:
        file.write(response.text)

    time.sleep(0.5)

def scrape_slides(lecture_id):
    url = f'https://cdn.cs50.net/ai/2020/spring/lectures/{lecture_id}/lecture{lecture_id}.pdf'
    
    if not os.path.exists(f'./scraped_data/lecture_slides/lecture_{lecture_id}'):
        os.makedirs(f'./scraped_data/lecture_slides/lecture_{lecture_id}')
        
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(f"./scraped_data/lecture_slides/lecture_{lecture_id}/slides.pdf", 'wb') as f:
            f.write(response.content)
        print(f"Lecture {lecture_id} PDF downloaded successfully.")
    else:
        print(f"Failed to download lecture {lecture_id} PDF. Status code: {response.status_code}")
    
    time.sleep(0.5)
    
def scrape_subs(lecture_id):
    response = requests.get(f'https://cdn.cs50.net/ai/2020/spring/lectures/{lecture_id}/lang/en/lecture{lecture_id}.srt')

    if not os.path.exists(f'./scraped_data/subtitles/lecture_{lecture_id}'):
        os.makedirs(f'./scraped_data/subtitles/lecture_{lecture_id}')

    with open(f'./scraped_data/subtitles/lecture_{lecture_id}/subtitles.txt', 'w') as file:
        file.write(response.text)

    time.sleep(0.5)
    

def scrape_code(lecture_id):
    response = requests.get(f'https://cdn.cs50.net/ai/2020/spring/lectures/{lecture_id}/src{lecture_id}.zip')
    
    if not os.path.exists(f'./scraped_data/source_code/lecture_{lecture_id}'):
        os.makedirs(f'./scraped_data/source_code/lecture_{lecture_id}')
    
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(f'./scraped_data/source_code/lecture_{lecture_id}')
    else:
        print(f'Failed to download the file. Status code: {response.status_code}')   
    
    time.sleep(0.5)

if __name__ == '__main__':
    for lecture_id in range(7):
        scrape_notes(lecture_id)
        scrape_transcripts(lecture_id)
        scrape_slides(lecture_id)
        scrape_subs(lecture_id)
        scrape_code(lecture_id)

    if not os.path.exists(f'./scraped_data/quiz'):
        os.makedirs(f'./scraped_data/quiz')

    git.Repo.clone_from('https://github.com/wbsth/cs50ai', './scraped_data/quiz/')
