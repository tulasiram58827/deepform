from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import pdfplumber
import wandb
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm import tqdm

from deepform.common import PDF_DIR, S3_BUCKET
from deepform.document import SINGLE_CLASS_PREDICTION
from deepform.logger import logger
from deepform.util import docrow_to_bbox, dollar_match, wandb_bbox


def get_pdf_path(slug):
    """Return a path to the pdf with the given slug, downloading the file if necessary.

    If the pdf isn't in the local file system, download it from an external repository.
    """
    filename = slug + ("" if slug.endswith(".pdf") else ".pdf")
    location = PDF_DIR / filename
    if not location.is_file():
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        download_from_remote(location)
    return location


def get_pdf_paths(slugs):
    with ThreadPoolExecutor() as executor:
        print(f"Getting {len(slugs):,} pdfs...")
        for path in tqdm(executor.map(get_pdf_path, slugs), total=len(slugs)):
            yield path


def download_from_remote(local_path):
    """Copy a pdf from S3 into the local filesystem."""
    filename = local_path.name
    s3_key = "pdfs/" + filename
    s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
    try:
        s3.Bucket(S3_BUCKET).download_file(s3_key, str(local_path))
    except ClientError:
        logger.error(f"Unable to retrieve {s3_key} from s3://{S3_BUCKET}")
        raise

def log_pdfs(doc, doc_log, all_scores):
    fname = get_pdf_path(doc.slug)
    try:
        pdf = pdfplumber.open(fname)
    except Exception:
        # If the file's not there, that's fine -- we use available PDFs to
        # define what to see
        print(f"Cannot open pdf {fname}")
        return

    print(f"Rendering output for {fname}")

    # Get the correct answers: find the indices of the token(s) labelled 1
    target_idx = [idx for (idx, val) in enumerate(doc.labels) if val == 1]

    class_ids_by_field = {"gross_amount" : 0, "flight_to" : 1, "flight_from" : 2, "contract_num" : 3, "advertiser" : 4}
    class_id_to_label = {int(v): k for k, v in class_ids_by_field.items()}

    # Draw the machine output: get a score for each token
    page_images = []
    for pagenum, page in enumerate(pdf.pages):

        im = page.to_image(resolution=300)
        #print("PAGE w h: ", page.width, " ",  page.height)
        #print("ORIG SIZE: ", im.original.size)
        #print("ROOT SIZE: ", im.root.width, " ", im.root.height)
        
        # training data has 0..1 for page range (see create-training-data.py)
        num_pages = len(pdf.pages)
        if num_pages > 1:
            current_page = pagenum / float(num_pages - 1)
        else:
            current_page = 0.0
        print("TOTAL PAGES: ", num_pages)
        print("THIS PAGE: ", current_page)
        # Draw guesses
        # loop over all predictions
        pred_bboxes = []
        true_bboxes = []
        field_colors = {"gross_amount" : "magenta", "flight_to" : "red", "flight_from" : "green", "advertiser" : "blue", "contract_num" : "orange"} 
        for i, score in enumerate(doc_log["score"]):
            rel_score = all_scores[:, i] / score
            page_match = np.isclose(doc.tokens["page"], current_page)
            curr_field = doc_log["field"][i]
            fc = field_colors[curr_field]
            for token in doc.tokens[page_match & (rel_score > 0.9)].itertuples():
                if rel_score[token.Index] == 1:
                    w = 8
                    #s = "magenta"
                elif rel_score[token.Index] >= 0.75:
                    w = 4
                    #s = "red"
                else:
                    w = 1
                    #s = "red"
                im.draw_rect(docrow_to_bbox(token), stroke="magenta", stroke_width=w, fill=None)
                pred_bboxes.append(wandb_bbox(token, rel_score[token.Index], class_ids_by_field[curr_field], im, page.width, page.height))
            # Draw target tokens
            target_toks = [
                doc.tokens.iloc[i]
                for i in target_idx
                if np.isclose(doc.tokens.iloc[i]["page"], current_page)
            ]
            rects = [docrow_to_bbox(t) for t in target_toks]
            # TODO: what is the field for the target tokens?
            true_bboxes.extend([wandb_bbox(t, 1, class_ids_by_field[curr_field], im, page.width, page.height) for t in target_toks]) 
            #im.draw_rects(rects, stroke="yellow", stroke_width=4, fill=None)
       
        boxes = {"predictions" : { "box_data" : pred_bboxes, "class_labels" : class_id_to_label}, \
                 "ground_truth" : {"box_data" : true_bboxes, "class_labels" : class_id_to_label}}
        #page_images.append(wandb.Image(im.annotated, boxes=boxes, caption="page " + str(pagenum)))
        page_images.append(wandb.Image(im.annotated, boxes=boxes, caption=fname.name + " p" + str(pagenum)))

        # get best matching score of any token in the training data
    #match = doc.tokens[SINGLE_CLASS_PREDICTION].max()
    #caption = (
    #    f"{doc.slug} guessed:{predict_text} answer:{answer_text} match:{match:.2f}"
    #)
   # verdict = dollar_match(predict_text, answer_text)
    wandb.log({"docs": page_images}) 
    #return verdict, caption, page_images

def log_pdf(doc, score, scores, predict_text, answer_text):
    fname = get_pdf_path(doc.slug)
    try:
        pdf = pdfplumber.open(fname)
    except Exception:
        # If the file's not there, that's fine -- we use available PDFs to
        # define what to see
        print(f"Cannot open pdf {fname}")
        return

    print(f"Rendering output for {fname}")

    # Get the correct answers: find the indices of the token(s) labelled 1
    target_idx = [idx for (idx, val) in enumerate(doc.labels) if val == 1]

    # Draw the machine output: get a score for each token
    page_images = []
    for pagenum, page in enumerate(pdf.pages):
        im = page.to_image(resolution=300)

        # training data has 0..1 for page range (see create-training-data.py)
        num_pages = len(pdf.pages)
        if num_pages > 1:
            current_page = pagenum / float(num_pages - 1)
        else:
            current_page = 0.0

        # Draw guesses
        rel_score = scores / score
        page_match = np.isclose(doc.tokens["page"], current_page)
        for token in doc.tokens[page_match & (rel_score > 0.5)].itertuples():
            if rel_score[token.Index] == 1:
                w = 5
                s = "magenta"
            elif rel_score[token.Index] >= 0.75:
                w = 3
                s = "red"
            else:
                w = 1
                s = "red"
            im.draw_rect(docrow_to_bbox(token), stroke=s, stroke_width=w, fill=None)

        # Draw target tokens
        target_toks = [
            doc.tokens.iloc[i]
            for i in target_idx
            if np.isclose(doc.tokens.iloc[i]["page"], current_page)
        ]
        rects = [docrow_to_bbox(t) for t in target_toks]
        im.draw_rects(rects, stroke="blue", stroke_width=3, fill=None)

        page_images.append(wandb.Image(im.annotated, caption="page " + str(pagenum)))

    # get best matching score of any token in the training data
    match = doc.tokens[SINGLE_CLASS_PREDICTION].max()
    caption = (
        f"{doc.slug} guessed:{predict_text} answer:{answer_text} match:{match:.2f}"
    )
    verdict = dollar_match(predict_text, answer_text)
    return verdict, caption, page_images
