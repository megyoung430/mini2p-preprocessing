"""Use exact embedded green images from the original high-resolution PDF."""
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import re

import numpy as np
from PIL import Image
from pypdf import PdfReader
from pypdf.generic import ContentStream

ROOT = Path('/Volumes/behrens/meg/3x3_field_magnitude_bandit/derivatives/cohort-01/sub-03_id-MY_71_R')
SOURCE = ROOT / 'meanImg_high_resolution_136_pages_sessions-216-231.pdf'


@lru_cache(maxsize=1)
def source_pages():
    # Read once; avoid repeated small reads from the network drive.
    reader = PdfReader(BytesIO(SOURCE.read_bytes()))
    pages = {}
    for page in reader.pages:
        text = page.extract_text()
        if 'Green functional' not in text:
            continue
        session, run = re.search(r'Session (\d+) \| (run-\d+)', text).groups()
        key = (session, run)
        assert key not in pages
        pages[key] = page
    assert len(pages) == 68
    return reader, pages


def load_mean_image(plane):
    reader, pages = source_pages()
    session = plane.parents[3].name.split('_')[0].removeprefix('ses-')
    run = plane.parents[1].name
    page = pages[(session, run)]
    # Matplotlib shares all image resources across pages; use only the image
    # actually drawn by this page's Do operator, not page.images[0].
    names = [args[0] for args, op in ContentStream(page.get_contents(), reader).operations
             if op == b'Do']
    assert len(names) == 1
    return page.images[names[0]].image.convert('RGB')


def panel(plane):
    image = load_mean_image(plane)
    seg = plane / 'meanImg_green_functional_contrast_seg.npy'
    if not seg.exists():
        return image, 0
    masks = np.asarray(np.load(seg, allow_pickle=True).item()['masks'])
    assert masks.ndim == 2
    # The source PDF contains nearest-neighbour upscaling of the native image.
    # Check the native image dimensions before applying the same scaling to masks.
    with Image.open(plane / 'meanImg_green_functional_contrast.png') as native:
        assert masks.shape == (native.height, native.width)
        assert abs(image.width/image.height - native.width/native.height) < .001
    padded = np.pad(masks, 1)
    edges = (masks > 0) & (
        (masks != padded[:-2, 1:-1]) | (masks != padded[2:, 1:-1]) |
        (masks != padded[1:-1, :-2]) | (masks != padded[1:-1, 2:]))
    edges = np.asarray(Image.fromarray(edges).resize(image.size, Image.Resampling.NEAREST))
    rgb = np.array(image)
    rgb[edges] = [40, 235, 255]
    count = int(np.count_nonzero(np.unique(masks) > 0))
    return Image.fromarray(rgb), count
