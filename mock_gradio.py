# Mock gradio module for testing
class MockGradio:
    class Error(Exception):
        pass

    class Markdown:
        def __init__(self, *args, **kwargs):
            pass

    class Textbox:
        def __init__(self, *args, **kwargs):
            pass

    class Slider:
        def __init__(self, *args, **kwargs):
            pass

    class Checkbox:
        def __init__(self, *args, **kwargs):
            pass

    class Button:
        def __init__(self, *args, **kwargs):
            pass

    class Dataframe:
        def __init__(self, *args, **kwargs):
            pass

    class Gallery:
        def __init__(self, *args, **kwargs):
            pass

    class Dropdown:
        def __init__(self, *args, **kwargs):
            pass

    @staticmethod
    def update(*args, **kwargs):
        return {}

# Create an instance to assign to gr
gr = MockGradio()

# Also expose the classes and functions at module level for direct import
Error = MockGradio.Error
Markdown = MockGradio.Markdown
Textbox = MockGradio.Textbox
Slider = MockGradio.Slider
Checkbox = MockGradio.Checkbox
Button = MockGradio.Button
Dataframe = MockGradio.Dataframe
Gallery = MockGradio.Gallery
Dropdown = MockGradio.Dropdown
update = MockGradio.update