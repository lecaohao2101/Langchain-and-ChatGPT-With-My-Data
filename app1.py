import os
import dotenv
import openai
from pymongo import MongoClient
from cloudinary import CloudinaryVideo

# Import API key from .env file
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.transcripts
transcript_collection = db.transcripts

# Cloudinary credentials
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

# Cloudinary video object
cloudinary_video = CloudinaryVideo(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)

# Speech to text
def transcribe(input_file):
    transcript = openai.Audio.transcribe("whisper-1", input_file)
    return transcript

# Crawl and transcribe videos
def crawl_and_transcribe():
    # Get all videos from Cloudinary
    videos = cloudinary_video.resources(resource_type="video")

    for video in videos:
        # Download video file
        video_file = cloudinary_video.download(video.public_id)

        # Transcribe video
        transcript_text = transcribe(video_file)["text"]

        # Save transcript text in MongoDB with the same filename as the video
        transcript_file_name = f"{video.public_id}_transcript.txt"
        transcript_collection.insert_one({"name": transcript_file_name, "text": transcript_text})

if __name__ == "__main__":
    crawl_and_transcribe()