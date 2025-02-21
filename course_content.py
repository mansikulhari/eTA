import os, io, git
from utils import get_text_chunks
from old_scrape import scrape_notes, scrape_transcripts, scrape_slides, scrape_subs, scrape_code


class CourseContent:
    def __init__(self, course_id, course_name, needs_scraping=False):
        if needs_scraping:
            self.scrape_data()

        self.course_id = course_id
        self.course_name = course_name
        self.lecture_notes = []
        self.images = []
        self.transcripts = []
        self.parse_lecture_notes()
        self.collect_images()
        self.parse_transcripts()

    def scrape_data(self):
        for lecture_id in range(7):
            scrape_notes(lecture_id)
            scrape_transcripts(lecture_id)
            scrape_slides(lecture_id)
            scrape_subs(lecture_id)
            scrape_code(lecture_id)

        if not os.path.exists(f'./scraped_data/quiz'):
            os.makedirs(f'./scraped_data/quiz')

        git.Repo.clone_from('https://github.com/wbsth/cs50ai', './scraped_data/quiz/')

    def parse_lecture_notes(self):
        base_dir = 'scraped_data/notes'

        all_notes = []
        for i in range(7):
            # Construct the path to the notes.txt file for each lecture
            lecture_dir = os.path.join(base_dir, f'lecture_{i}')
            notes_path = os.path.join(lecture_dir, 'notes.txt')

            if os.path.exists(notes_path):
                with open(notes_path, 'r') as file:
                    content = file.read()
                    text_chunks = get_text_chunks(content, chunk_size=75, chunk_overlap=5)
                    all_notes.extend(text_chunks)
            else:
                print(f"No notes.txt found in {lecture_dir}")

        self.lecture_notes = all_notes

    def collect_images(self):
        base_dir = 'scraped_data/notes'
        all_images = []
        for i in range(7):
            lecture_dir = os.path.join(base_dir, f'lecture_{i}')

            images = []
            images_dir = os.path.join(lecture_dir, 'images')

            # Check if the images directory exists
            if not os.path.exists(images_dir):
                print(f"No images directory found in {lecture_dir}")
                continue

            # Iterate through the files in the images directory
            for filename in os.listdir(images_dir):
                # Construct the full file path
                file_path = os.path.join(images_dir, filename)

                # Check if the file is an image by checking its extension
                if file_path.lower().endswith('.png'):
                    # Open and append the image to the list
                    with open(file_path, 'rb') as img_file:
                        img_byte_arr = io.BytesIO(img_file.read())
                        images.append(img_byte_arr)

            all_images.extend(images)

        self.images = all_images

    def parse_transcripts(self):
        base_dir = 'scraped_data/transcript'
        all_transcripts = []
        for i in range(7):
            transcript_dir = os.path.join(base_dir, f'lecture_{i}/transcript.txt')
            if os.path.exists(transcript_dir):
                with open(transcript_dir, 'r') as file:
                    content = file.read()
                    text_chunks = get_text_chunks(content, chunk_size=75, chunk_overlap=5)
                    all_transcripts.extend(text_chunks)

        self.transcripts = all_transcripts


    def get_all_text(self):
        return self.lecture_notes + self.transcripts

    def get_all_images(self):
        return self.images

