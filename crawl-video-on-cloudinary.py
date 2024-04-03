import cloudinary.utils
import cloudinary
import cloudinary.api
import requests
import os
import dotenv

dotenv.load_dotenv()
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)

def list_videos(next_cursor=None):
    result = cloudinary.api.resources(
        resource_type="video",
        type="upload",
        max_results=500,
        next_cursor=next_cursor
    )
    return result

def download_video(url, filename):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)

def remove_duplicates(video_urls):
    unique_urls = []
    seen_filenames = set()

    for url in video_urls:
        filename = url.split('/')[-1]
        if filename not in seen_filenames:
            unique_urls.append(url)
            seen_filenames.add(filename)

    return unique_urls

download_dir = "downloaded_videos"
os.makedirs(download_dir, exist_ok=True)

next_cursor = None
all_videos = []

while True:
    result = list_videos(next_cursor)
    all_videos.extend(result['resources'])

    if 'next_cursor' in result:
        next_cursor = result['next_cursor']
    else:
        break

video_urls = [video['secure_url'] for video in all_videos]
unique_urls = remove_duplicates(video_urls)

for url in unique_urls:
    filename = url.split('/')[-1]
    filepath = os.path.join(download_dir, filename)

    if os.path.exists(filepath):
        continue

    download_video(url, filepath)
    print(f"Downloaded: {filename}")

print("All unique videos downloaded!")