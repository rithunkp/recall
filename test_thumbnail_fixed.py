import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))

# Mock gradio before importing app
class MockGradio:
    class Error(Exception):
        pass

    @staticmethod
    def update(*args, **kwargs):
        return {}

# Replace the gradio module with our mock
sys.modules['gradio'] = MockGradio()

# Now we can import the app
from demo.app import get_thumbnail_path
import os

# Test with the existing smoke memory frame
frame_path = "data\\frames\\UCSDped1\\Train013\\106.tif"
thumb = get_thumbnail_path(frame_path)
print(f"Thumbnail path: {thumb}")
print(f"Exists: {os.path.exists(thumb)}")
# Check if it's a PNG
if thumb.lower().endswith('.png'):
    print("Thumbnail is PNG")
else:
    print("Thumbnail is not PNG")