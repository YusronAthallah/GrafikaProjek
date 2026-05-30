import pygame
import numpy as np
import math
import os

SAMPLE_RATE = 44100
DURATION = 3.0

def _generate_buffer(func, duration=DURATION, volume=0.25):
    """Generate a stereo sound buffer from a waveform function."""
    n_samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)
    wave = func(t) * volume
    wave = np.clip(wave, -1.0, 1.0)
    samples = (wave * 32767).astype(np.int16)
    stereo = np.column_stack((samples, samples))
    sound = pygame.sndarray.make_sound(stereo)
    return sound


class SoundManager:
    """Manages procedural audio mapped to gesture modes."""
    
    def __init__(self, sound_dir=None):
        # Initialize mixer with specific settings for procedural audio
        pygame.mixer.pre_init(SAMPLE_RATE, -16, 2, 2048)
        # Note: pygame.mixer.init() should be called after pygame.init()
        self.sounds = {}
        self.current_mode = None
        self.enabled = False
        self.initialized = False
        self._channel = None
        # Directory tempat file suara berada
        self.sound_dir = sound_dir or os.path.dirname(os.path.abspath(__file__))
    
    def init(self):
        """Generate all sound buffers. Call AFTER pygame.init()."""
        if self.initialized:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(SAMPLE_RATE, -16, 2, 2048)
            
            self.sounds = {}
            
            # Mode 5: Load hidup-jokowi.mp3 dari file
            mp3_path = os.path.join(self.sound_dir, "hidup-jokowi.mp3")
            if os.path.exists(mp3_path):
                self.sounds[5] = pygame.mixer.Sound(mp3_path)
                self.sounds[5].set_volume(0.5)
                print(f"[SoundManager] Loaded: {mp3_path}")
            else:
                print(f"[SoundManager] WARNING: {mp3_path} tidak ditemukan, audio untuk mode ini dinonaktifkan.")
            
            # Mode 6: Load jokowi-saya-akan-lawan.mp3 dari file
            lawan_path = os.path.join(self.sound_dir, "jokowi-saya-akan-lawan.mp3")
            if os.path.exists(lawan_path):
                self.sounds[6] = pygame.mixer.Sound(lawan_path)
                self.sounds[6].set_volume(0.5)
                print(f"[SoundManager] Loaded: {lawan_path}")
            else:
                print(f"[SoundManager] WARNING: {lawan_path} tidak ditemukan, audio untuk mode ini dinonaktifkan.")
            
            self._channel = pygame.mixer.Channel(0)
            self.initialized = True
            print("[SoundManager] Initialized successfully.")
        except Exception as e:
            print(f"[SoundManager] Init error: {e}")
            self.initialized = False
    
    def toggle(self):
        """Toggle sound on/off. Returns new state."""
        self.enabled = not self.enabled
        if not self.enabled:
            self.stop()
        return self.enabled
    
    def update(self, mode):
        """Update playback based on current gesture mode."""
        if not self.enabled or not self.initialized:
            if self._channel and self._channel.get_busy():
                self._channel.stop()
            return
        
        if mode != self.current_mode:
            self.current_mode = mode
            self._play_mode(mode)
    
    def _play_mode(self, mode):
        """Play the sound associated with a mode, looping."""
        if self._channel:
            if mode in self.sounds:
                self._channel.stop()
                self._channel.play(self.sounds[mode], loops=-1, fade_ms=500)
            else:
                # Mode tanpa suara: hentikan suara sebelumnya
                self._channel.fadeout(300)
    
    def stop(self):
        """Stop all sounds."""
        if self._channel:
            self._channel.fadeout(300)
        self.current_mode = None
    
    def cleanup(self):
        """Clean up resources."""
        self.stop()
        self.sounds.clear()