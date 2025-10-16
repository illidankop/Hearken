# Import necessary libraries
import os
import shutil
import numpy as np
import soundfile as sf

# Define a constant for the buffer size
BUFFER_SIZE = 30

# Function to ensure a directory is empty
def ensure_empty_directory(path):
    # If the directory exists, remove it and all its contents
    if os.path.exists(path):
        shutil.rmtree(path)
    # Create a new empty directory
    os.makedirs(path)

# Function to read audio file into frames of fixed duration
def read_audio_to_frames(input_path, frame_duration=1):
    # Open the audio file
    with sf.SoundFile(input_path) as sound_file:
        sample_rate = sound_file.samplerate  # Get the sample rate
        num_channels = sound_file.channels  # Get the number of channels
        frame_length = sample_rate * frame_duration  # Calculate the length of each frame
        
        # Loop to read the audio file in chunks
        while True:
            frames = sound_file.read(frame_length)  # Read a chunk of audio data
            if len(frames) == 0:  # If no more data, exit the loop
                break
            # If the chunk is smaller than the desired frame length, pad it with zeros
            if frames.shape[0] < frame_length:
                # Create an array of zeros with the same number of channels and the missing frame length
                padding = np.zeros((frame_length - frames.shape[0], num_channels))
                # Concatenate the frames and padding vertically
                frames = np.vstack((frames, padding))
            # Yield the frames to the caller
            yield frames

# Class to manage audio buffer
class AudioBufferManager:
    def __init__(self, buffer_size=BUFFER_SIZE, sample_rate=44100, num_channels=1):
        self.buffer_size = buffer_size * sample_rate  # Total buffer size in samples
        # Initialize an empty buffer with no rows and num_channels columns
        self.buffer = np.empty((0, num_channels))
        self.is_initialized = False  # Flag to check if the buffer is filled
        self.last_frame_time = None  # Track the time of the last frame

    def manage_audio_buffer(self, frame, frame_time):
        # Check if the frame is new
        if self.last_frame_time is not None and frame_time <= self.last_frame_time:
            yield None, False  # No new frame to process
            return
        # Update the last frame time
        self.last_frame_time = frame_time
        
        # Concatenate the new frame to the buffer vertically
        self.buffer = np.vstack((self.buffer, frame))

        # If buffer is not yet filled, set is_initialized to True
        if not self.is_initialized:
            # Check if the buffer has reached the desired size
            if self.buffer.shape[0] >= self.buffer_size:
                self.is_initialized = True
        else:
            # If buffer is already filled, add the frame and keep only the latest data
            if self.buffer.shape[0] > self.buffer_size:
                # Keep only the last buffer_size samples
                self.buffer = self.buffer[-self.buffer_size:]
        
        # Yield the buffer if it's initialized, otherwise yield None
        if self.is_initialized:
            yield self.buffer, True
        else:
            yield None, False
            