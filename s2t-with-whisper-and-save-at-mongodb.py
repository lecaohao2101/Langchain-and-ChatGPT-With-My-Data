import cloudinary.utils
import cloudinary
import cloudinary.api
import requests
import os
import dotenv
from pymongo import MongoClient
import openai
import whisper

# Load environment variables
dotenv.load_dotenv()
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
MONGO_URI = os.getenv("MONGO_URI")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set Cloudinary configuration
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)

# MongoDB connection
client = MongoClient(MONGO_URI)
db = client.transcripts
transcript_collection = db.transcripts

# Define functions
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

def transcribe_video(video_file_path):
    # Load the Whisper model
    model = whisper.load_model("base")

    # Transcribe the video file
    transcript = model.transcribe(video_file_path)

    # Extract the transcript text
    transcript_text = transcript["text"]

    return transcript_text

def main():
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

    # Get existing transcript names from MongoDB
    existing_transcript_names = set(transcript['name'] for transcript in transcript_collection.find({}, {'name': 1, '_id': 0}))

    for url in unique_urls:
        filename = url.split('/')[-1]
        filepath = os.path.join(download_dir, filename)

        # Check if video file already exists in downloaded_videos folder
        if os.path.exists(filepath):
            print(f"Skipping {filename} (already downloaded)")
            continue

        download_video(url, filepath)
        print(f"Downloaded: {filename}")

        # Transcribe video to text
        transcript_text = transcribe_video(filepath)

        # Check if transcript already exists in MongoDB
        transcript_file_name = f"{os.path.splitext(filename)[0]}.txt"
        if transcript_file_name in existing_transcript_names:
            print(f"Skipping {transcript_file_name} (already exists in MongoDB)")
            continue

        # Save transcript text in MongoDB
        transcript_collection.insert_one({"name": transcript_file_name, "text": transcript_text})
        print(f"Transcript for {filename} saved in MongoDB as {transcript_file_name}")

if __name__ == "__main__":
    main()