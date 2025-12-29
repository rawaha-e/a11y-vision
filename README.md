# a11y-vision
A proof of concept for multimodal approach to generate accessible webpages for use with screen readers when accessibility markers are not present.

## The idea
Segment a webpage screenshot into components, map them to DOM elements, and generate an annotated, accessible view.

## Current status
The project is in a PoC staghe, your contributions are welcome to bring it to life as a viable software package!

#### TODO
- Generate annotations for segmented content
- Link existing webpage elements to an acessibility tree
- Create a web extension to automate the screen capture and UI

## Setup
1. Clone the repository:
```bash
git clone https://github.com/rawaha-e/a11y-vision
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Download SAM2 checkpoint:
```bash
mkdir checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O checkpoints/sam2.1_hiera_large.pt
```

4. Place `screenshot.png` in the project directory.

5. Run the inference:
```bash
python3 segment_webpage.py
```