from decimal import Decimal
from math import isclose
from pathlib import Path

import pandas as pd
import pdfplumber

PDF_PATH = Path("../data/pdfs/")
IMG_PATH = Path("./pages")
MASK_PATH = Path("./masks")

BBOX_MIN_HEIGHT = 10
BBOX_MIN_WIDTH = 40


def docrow_to_bbox(t):
    bbox = [
        Decimal(t["x0"]) - 1,
        Decimal(t["y0"]) - 1,
        Decimal(t["x1"]) + 1,
        Decimal(t["y1"]) + 1,
    ]
    # Hacky fix for bboxes that underline rather than bound the token.
    bbox[1] = min(bbox[3] - BBOX_MIN_HEIGHT, bbox[1])
    bbox[0] = min(bbox[2] - BBOX_MIN_WIDTH, bbox[0])
    return bbox


def slug_to_img_filename(slug, page=1):
    if slug.endswith(".pdf"):
        slug = slug[:-4]
    return f"{slug}_p{page:02}.png"


def render_first_page_image_and_mask(slug, bboxes):
    # print(f"Adding {len(bboxes)} boxes to {slug}")
    pdf_filepath = PDF_PATH / slug
    img_filename = slug_to_img_filename(slug, page=1)
    try:
        pdf = pdfplumber.open(pdf_filepath)
    except FileNotFoundError:
        return

    first_page = pdf.pages[0]
    im = first_page.to_image(resolution=300)
    im.save(IMG_PATH / img_filename, format="PNG")

    blank = [0, 0, 770, 780]
    im.draw_rect(blank, fill="white", stroke_width=0)
    im.draw_rects(bboxes, fill="black", stroke_width=0)
    im.save(MASK_PATH / img_filename, format="PNG")


def create_masks_from_pdfs():
    training_data = pd.read_parquet("../source/training.parquet")

    token_matches = training_data[training_data["gross_amount"] == 1]

    for slug, group in token_matches.groupby("slug"):
        bboxes = group[group["page"] == 0].apply(docrow_to_bbox, axis=1)
        if not bboxes.empty:
            render_first_page_image_and_mask(slug, bboxes)


if __name__ == "__main__":
    create_masks_from_pdfs()
