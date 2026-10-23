"""Create a landscape run-by-run Cellpose curation comparison PDF."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader

ROOT = Path('/Volumes/behrens/meg/3x3_field_magnitude_bandit/derivatives/cohort-01/sub-03_id-MY_71_R')
OUT = Path(__file__).resolve().parents[1] / 'output/pdf/71R_sessions_216-231_curation_reference_219_run05.pdf'
IMAGE = 'meanImg_green_functional_contrast.png'
SEG = 'meanImg_green_functional_contrast_seg.npy'


from curation_source_images import load_mean_image, panel


def main():
    planes = [p for s in range(216, 232)
              for p in sorted(ROOT.glob(f'ses-{s}_*/funcimg/run-*/suite2p/plane0'))]
    missing = [str(p) for p in planes if not (p / IMAGE).exists()]
    if missing:
        raise FileNotFoundError(f'Missing green images: {missing}')
    refs = list(ROOT.glob('ses-219_*/funcimg/run-05/suite2p/plane0'))
    assert len(refs) == 1
    reference, ref_count = panel(refs[0])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(OUT), pagesize=landscape(A4), pageCompression=1)
    pdf.setTitle('71R | Segmentation curation | Sessions 216-231')
    pdf.setAuthor('Mini2p preprocessing')
    w, h = landscape(A4)
    margin, gap = 28, 24
    col = (w - 2 * margin - gap) / 2
    records = []
    for page, plane in enumerate(planes, 1):
        session = plane.parents[3].name
        run = plane.parents[1].name
        session_number = session.split('_')[0].removeprefix('ses-')
        date = session.split('date-')[1]
        current, count = panel(plane)
        pdf.setFillColorRGB(.10, .14, .18)
        pdf.setFont('Helvetica-Bold', 17)
        pdf.drawString(margin, h - 34, '71R / Segmentation curation')
        pdf.setFont('Helvetica', 9)
        pdf.setFillColorRGB(.35, .4, .45)
        pdf.drawRightString(w-margin, h-33, f'Sessions 216-231  |  {page} / {len(planes)}')
        for x, img, title, subtitle, n in [
            (margin, current, f'Session {session_number} / {run}', f'{date[:4]}-{date[4:6]}-{date[6:]}  |  Review', count),
            (margin+col+gap, reference, 'Session 219 / run-05', '2026-08-02  |  Fixed reference', ref_count),
        ]:
            pdf.setFillColorRGB(.10, .14, .18)
            pdf.setFont('Helvetica-Bold', 13)
            pdf.drawString(x, h-69, title)
            pdf.setFont('Helvetica', 9)
            pdf.setFillColorRGB(.35, .4, .45)
            status = f'{n} cells' if n else 'No outlines'
            pdf.drawString(x, h-85, f'{subtitle}  |  {status}')
            scale = min(col / img.width, (h - 145) / img.height)
            iw, ih = img.width*scale, img.height*scale
            pdf.drawImage(ImageReader(img), x+(col-iw)/2, h-100-ih, width=iw, height=ih)
        pdf.setStrokeColorRGB(.82,.85,.88)
        pdf.line(w/2, 45, w/2, h-57)
        pdf.setFont('Helvetica', 8)
        pdf.setFillColorRGB(.35,.4,.45)
        pdf.drawString(margin, 26, 'Green image from original high-resolution PDF | Cyan: cell boundaries | No registration')
        pdf.showPage()
        records.append(dict(page=page, session=session, run=run, cells=count, plane=str(plane)))
    pdf.save()
    Path('tmp/pdfs/71R_manifest.json').write_text(json.dumps(records, indent=2))
    print(f'{OUT}\n{len(records)} pages; {sum(r["cells"] == 0 for r in records)} without outlines')


if __name__ == '__main__':
    main()
