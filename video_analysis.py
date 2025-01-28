import json
import os
from typing import Tuple
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import whisper
import torch
print(torch.cuda.is_available()) 

class Video:
    language = "en"
    summary = ""
    transcript = []
    embeddings = []



class VideoAnalyzer_old:
    def __init__(self, video_path):
        self.video_path = video_path
        self.model = WhisperModel("base", device="cuda")
        self.nlp_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(384)  # Assuming embeddings are 384-dimensional

    def transcribe_video(self):
        segments, _ = self.model.transcribe(self.video_path, beam_size=5)
        transcript = []
        for segment in segments:
            transcript.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
        return transcript

    def generate_embeddings(self, transcript):
        embeddings = [self.nlp_model.encode(section["text"]) for section in transcript]
        return embeddings

    def create_faiss_index(self, embeddings):
        self.index.add(np.array(embeddings))

    def query_video(self, query):
        query_embedding = self.nlp_model.encode(query)
        _, indices = self.index.search(np.array([query_embedding]), 1)
        return indices[0][0]

class VideoAnalyzer:
    def __init__(self, video_path: str, device: str = "cuda"):
        self.video_path = video_path
        self.device = device
        self.transcript = []
        self.embeddings = []
        self.refined_segments = []
        
        # Models
        self.whisper_model = WhisperModel("base", device=self.device)
        self.nlp_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")  # Better embedding model
        self.faiss_index = None

    def transcribe_video(self) -> list:
        """Transcribes the video into text segments."""
        segments, _ = self.whisper_model.transcribe(self.video_path, beam_size=5)
        self.transcript = [
            {"start": segment.start, "end": segment.end, "text": segment.text.strip()}
            for segment in segments
        ]
        return self.transcript

    def split_text_into_segments(self, chunk_size: int = 500, chunk_overlap: int = 20) -> list:
        """Splits transcribed text into chunks for embeddings."""
        combined_text = " ".join([seg["text"] for seg in self.transcript])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_segments = text_splitter.split_text(combined_text)

        refined_segments = []
        current_index = 0

        for split in split_segments:
            segment_start, segment_end = None, None
            accumulated_text = ""

            for i in range(current_index, len(self.transcript)):
                entry = self.transcript[i]
                if segment_start is None:
                    segment_start = entry["start"]

                accumulated_text += entry["text"] + " "
                segment_end = entry["end"]

                if split.strip() in accumulated_text.strip():
                    current_index = i + 1
                    break

            refined_segments.append({
                "text": split,
                "start": segment_start,
                "end": segment_end
            })

        self.refined_segments = refined_segments
        return self.refined_segments

    def generate_embeddings(self):
        """Generates embeddings for each segment."""
        if not self.refined_segments:
            raise ValueError("No refined segments found. Ensure `split_text_into_segments()` is called first.")
        self.embeddings = [self.nlp_model.encode(segment["text"], normalize_embeddings=True) for segment in self.refined_segments]
        return self.embeddings

    def create_faiss_index(self):
        """Creates and populates the FAISS index with embeddings."""
        if not self.embeddings:
            raise ValueError("No embeddings found. Ensure `generate_embeddings()` is called first.")
        
        embedding_dim = self.embeddings[0].shape[0]
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        self.faiss_index.add(np.array(self.embeddings))

    def query_video(self, query: str, k: int = 1) -> list:
        """Queries the FAISS index and returns the top-k matching segments."""
        if not self.faiss_index:
            raise ValueError("FAISS index is not created. Ensure `create_faiss_index()` is called first.")

        query_embedding = self.nlp_model.encode(query, normalize_embeddings=True)
        distances, indices = self.faiss_index.search(np.array([query_embedding]), k)
        results = [self.refined_segments[idx] for idx in indices[0]]
        return results
    def save_embeddings(self, save_path: str):
        """Saves embeddings, FAISS index, and refined segments to disk."""
        if not self.embeddings or not self.faiss_index or not self.refined_segments:
            raise ValueError("Ensure embeddings, FAISS index, and refined segments are generated before saving.")

        # Save refined segments as JSON
        with open(os.path.join(save_path, "refined_segments.json"), "w") as f:
            json.dump(self.refined_segments, f)

        # Save embeddings as NumPy array
        np.save(os.path.join(save_path, "embeddings.npy"), np.array(self.embeddings))

        # Save FAISS index
        faiss.write_index(self.faiss_index, os.path.join(save_path, "faiss_index.bin"))

    def load_embeddings(self, load_path: str):
        """Loads embeddings, FAISS index, and refined segments from disk."""
        # Load refined segments
        with open(os.path.join(load_path, "refined_segments.json"), "r") as f:
            self.refined_segments = json.load(f)

        # Load embeddings
        self.embeddings = np.load(os.path.join(load_path, "embeddings.npy"))

        # Load FAISS index
        self.faiss_index = faiss.read_index(os.path.join(load_path, "faiss_index.bin"))

def old_test():
    video_path = r'C:\Users\mmerl\projects\iahackaton\cleverbot\crawl4AI-examples\[Cedreo Tutorial] How to Add a Furniture.mp4'
    # Step 1: Transcribe video
    model = WhisperModel("base", device="cuda")
    model = whisper.load_model("turbo")
    result = model.transcribe(video_path)
    segments, _ = model.transcribe(video_path, beam_size=5)

    transcript = []
    for segment in segments:
        transcript.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
    # Combine all text into a single string
    combined_text = " ".join([seg["text"] for seg in transcript])



    # Configure the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,   # Maximum characters per chunk
        chunk_overlap=20   # Overlap between chunks to ensure context
    )

    # Split the combined text
    split_segments = text_splitter.split_text(combined_text)

    refined_segments = []
    current_index = 0

    for split in split_segments:
        segment_start, segment_end = None, None
        accumulated_text = ""

        # Accumulate transcript entries to match the current split
        for i in range(current_index, len(transcript)):
            entry = transcript[i]
            if segment_start is None:
                segment_start = entry["start"]  # Mark the start timestamp

            accumulated_text += entry["text"] + " "  # Accumulate text
            segment_end = entry["end"]  # Update the end timestamp

            # Check if the accumulated text contains the split
            if split.strip() in accumulated_text.strip():
                current_index = i + 1  # Move the index forward
                break

        refined_segments.append({
            "text": split,
            "start": segment_start,
            "end": segment_end
        })

# Example Usage
if __name__ == "__main__":
    video_path = r'C:\Users\mmerl\projects\iahackaton\cleverbot\crawl4AI-examples\[Cedreo Tutorial] How to Add a Furniture.mp4'
    video_url = "https://www.youtube.com/watch?v=w6SazyiiH24"
    embedding_path = r'C:\Users\mmerl\projects\iahackaton\cleverbot\crawl4AI-examples'
    analyzer = VideoAnalyzer(video_path)
    print("Video Path:", analyzer.video_path)
    # Step 1: Transcribe Video
    transcript = analyzer.transcribe_video()
    print("Transcript:", transcript[:2])  # Print first 2 segments for verification

    # Step 2: Split Text into Segments
    refined_segments = analyzer.split_text_into_segments()
    print("Refined Segments:", refined_segments[:2])  # Print first 2 segments

    # Step 3: Generate Embeddings
    embeddings = analyzer.generate_embeddings()

    # Step 4: Create FAISS Index
    analyzer.create_faiss_index()
    analyzer.save_embeddings(embedding_path)

    # Step 5: Query the Video
    query = "How can I place a furniture on the wall?"
    results = analyzer.query_video(query)

    # Step 6: Generate Timestamped Link
    best_match = results[0]
    timestamped_url = f"{video_url}&t={int(best_match['start'])}"
    print("Relevant Section:", timestamped_url)