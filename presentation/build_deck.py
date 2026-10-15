#!/usr/bin/env python3
"""
build_deck.py
Academic research presentation generator adhering strictly to institutional templates:
- Preserves and populates the master title slide (Slide 0)[cite: 1, 4].
- Removes sample / instruction slides[cite: 1].
- Replaces dummy footer tokens ('MUSTER MANN', 'Max Mustermann', 'Name', etc.) across
  Slide Masters, Layouts, and instantiated slides.
- Pins content slide titles cleanly to the top-left banner area with left alignment.
- Centered section dividers for clear talk pacing[cite: 3].
- Supports varied slide archetypes: split, full_image, comparison, table, text_only[cite: 3].
- Renders elegant wireframe mockup boxes when figure image files are not yet generated.
- In-line Markdown bold (**bold**) and paragraph spacing support.
"""

import argparse
from pathlib import Path
import re
import yaml
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE


def replace_token_in_shape(shape, token_map: dict):
    """Replaces text tokens within a single shape while preserving formatting."""
    if not shape.has_text_frame:
        return
    for p in shape.text_frame.paragraphs:
        for run in p.runs:
            for token, value in token_map.items():
                if token.lower() in run.text.lower():
                    # Case-insensitive replacement preserving casing where possible
                    pattern = re.compile(re.escape(token), re.IGNORECASE)
                    run.text = pattern.sub(value, run.text)


def sanitize_footers_and_masters(prs, token_map: dict):
    """Traverses Slide Masters, Layouts, and individual slides to eliminate dummy names/tokens."""
    # 1. Clean Slide Masters (where the background canvas / fixed footer lives)
    for master in prs.slide_masters:
        for shape in master.shapes:
            replace_token_in_shape(shape, token_map)

        # 2. Clean Layouts inherited from each Master
        for layout in master.slide_layouts:
            for shape in layout.shapes:
                replace_token_in_shape(shape, token_map)

    # 3. Clean every instantiated Slide
    for slide in prs.slides:
        for shape in slide.shapes:
            replace_token_in_shape(shape, token_map)


def format_slide_header(slide, title_text: str, section_label: str = None):
    """Pins slide titles cleanly to the top-left banner area with left alignment."""
    if slide.shapes.title:
        title_shape = slide.shapes.title

        # Lock position and dimensions to the top-left banner region
        title_shape.left = Inches(0.8)
        title_shape.top = Inches(0.45)
        title_shape.width = Inches(11.5)
        title_shape.height = Inches(0.8)

        # Force TOP vertical anchoring
        tf = title_shape.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.TOP
        tf.text = title_text

        # Force LEFT alignment and typographic hierarchy
        for p in tf.paragraphs:
            p.font.size = Pt(24)
            p.font.bold = True
            p.font.color.rgb = RGBColor(20, 25, 35)
            p.alignment = PP_ALIGN.LEFT

    # Update section indicator if present in the slide layout / footer
    if section_label:
        for shape in slide.shapes:
            if shape.has_text_frame and any(k in shape.name.lower() for k in ["kapitel", "footer", "kategorie"]):
                shape.text_frame.text = section_label


def add_formatted_bullets(text_frame, bullet_list: list, font_size_pt=15, space_after_pt=12):
    """Parses markdown bold syntax (**bold**) and applies clean paragraph spacing."""
    text_frame.clear()
    text_frame.word_wrap = True

    for i, line in enumerate(bullet_list):
        p = text_frame.paragraphs[0] if i == 0 else text_frame.add_paragraph()
        p.font.size = Pt(font_size_pt)
        p.space_after = Pt(space_after_pt)
        p.line_spacing = 1.15
        p.level = 0

        # Parse **bold** parts
        parts = re.split(r'(\*\*.*?\*\*)', line)
        for part in parts:
            if not part:
                continue
            run = p.add_run()
            if part.startswith("**") and part.endswith("**"):
                run.text = part[2:-2]
                run.font.bold = True
                run.font.color.rgb = RGBColor(10, 15, 25)
            else:
                run.text = part
                run.font.bold = False
                run.font.color.rgb = RGBColor(60, 65, 75)


def add_image_or_placeholder(slide, img_cfg: dict, default_left=1.0, default_top=1.6, default_w=11.33, default_h=4.8):
    """Renders real picture or a clean wireframe mockup placeholder box."""
    path_str = img_cfg.get("path", "")
    p_img = Path(path_str) if path_str else None

    left = Inches(img_cfg.get("left_in", default_left))
    top = Inches(img_cfg.get("top_in", default_top))
    width = Inches(img_cfg.get("width_in", default_w))
    height = Inches(img_cfg.get("height_in", default_h))

    # Real image exists
    if p_img and p_img.exists():
        slide.shapes.add_picture(str(p_img), left=left, top=top, width=width)
        return

    # Visual wireframe mockup
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(248, 249, 251)
    shape.line.color.rgb = RGBColor(205, 210, 220)
    shape.line.width = Pt(1.0)

    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE

    label_title = img_cfg.get("label", "📊 [Abbildung / Visualisierung]")
    p1 = tf.paragraphs[0]
    p1.text = label_title
    p1.font.size = Pt(13)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(80, 85, 100)
    p1.alignment = PP_ALIGN.CENTER

    p2 = tf.add_paragraph()
    p2.text = f"Datei: {path_str if path_str else 'experiments/figures/...'}"
    p2.font.size = Pt(9.5)
    p2.font.italic = True
    p2.font.color.rgb = RGBColor(140, 145, 160)
    p2.alignment = PP_ALIGN.CENTER


def add_styled_table(slide, table_spec: dict, left_in=1.0, top_in=1.8, width_in=11.33):
    """Generates clean academic benchmark tables."""
    headers = table_spec.get("headers", [])
    rows = table_spec.get("rows", [])
    num_rows = len(rows) + 1
    num_cols = len(headers)

    table_shape = slide.shapes.add_table(
        num_rows, num_cols, Inches(left_in), Inches(top_in), Inches(width_in), Inches(0.4 * num_rows)
    )
    table = table_shape.table

    # Format header row
    for col_idx, header in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.text = header
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(235, 238, 243)
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(12)
            p.font.color.rgb = RGBColor(20, 25, 35)

    # Format content rows
    for row_idx, row_data in enumerate(rows):
        for col_idx, val in enumerate(row_data):
            cell = table.cell(row_idx + 1, col_idx)
            cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(11)
                p.font.color.rgb = RGBColor(50, 55, 65)


def build_presentation(template_path: Path, spec_path: Path) -> Path:
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    meta = spec.get("metadata", {})
    output_path = Path(meta.get("output", "docs/presentation/output/deck.pptx"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    author = meta.get("author", "")
    supervisor = meta.get("supervisor", "")
    title = meta.get("title", "")
    pres_type = meta.get("presentation_type", "MA/BA Präsentation")

    token_map = {
        "Vortrag zur MA / BA mit dem Titel „Titel“": f"{title}",
        #"„Titel“": f"„{title}“",
        #"Titel": title,
        "MA/BA Präsentation": pres_type,
        "Betreuer Name": supervisor,
        "Max Mustermann": author,
        "Name": author,
    }

    prs = Presentation(template_path)

    # 1. Clean masters, layouts, and existing template slides upfront
    sanitize_footers_and_masters(prs, token_map)

    # 2. Update Title Slide (Slide 0)[cite: 1, 4]
    if len(prs.slides) > 0:
        for shape in prs.slides[0].shapes:
            replace_token_in_shape(shape, token_map)

    # 3. Remove sample and instruction slides (indices 1 and higher)[cite: 1]
    while len(prs.slides) > 1:
        rId = prs.slides._sldIdLst[1].rId
        prs.part.drop_rel(rId)
        del prs.slides._sldIdLst[1]

    LAYOUT_CONTENT = 1
    LAYOUT_SECTION = 2 if len(prs.slide_layouts) > 2 else 1

    # 4. Build each slide based on the specification
    for item in spec.get("slides", []):
        stype = item.get("type", "split")

        if stype == "title":
            continue

        # --- Section Divider Slide ---[cite: 3]
        elif stype == "section":
            slide = prs.slides.add_slide(prs.slide_layouts[LAYOUT_SECTION])
            if slide.shapes.title:
                title_shape = slide.shapes.title
                title_shape.top = Inches(2.8)
                tf = title_shape.text_frame
                tf.text = item.get("title", "")
                tf.vertical_anchor = MSO_ANCHOR.MIDDLE
                for p in tf.paragraphs:
                    p.font.size = Pt(32)
                    p.font.bold = True
                    p.alignment = PP_ALIGN.CENTER

        # --- Full Image Slide (broad curves, grids, or Gantt charts) ---[cite: 3]
        elif stype == "full_image":
            slide = prs.slides.add_slide(prs.slide_layouts[LAYOUT_CONTENT])
            format_slide_header(slide, item.get("title", ""), item.get("section"))
            for ph in list(slide.placeholders):
                if ph != slide.shapes.title:
                    ph.element.getparent().remove(ph.element)
            add_image_or_placeholder(
                slide, item.get("image", {}), default_left=1.0, default_top=1.5, default_w=11.33, default_h=5.0
            )

        # --- Comparison Slide (two side-by-side plots/diagrams) ---[cite: 3]
        elif stype == "comparison":
            slide = prs.slides.add_slide(prs.slide_layouts[LAYOUT_CONTENT])
            format_slide_header(slide, item.get("title", ""), item.get("section"))
            for ph in list(slide.placeholders):
                if ph != slide.shapes.title:
                    ph.element.getparent().remove(ph.element)
            images = item.get("images", [])
            for i, img_cfg in enumerate(images):
                def_left = 1.0 + (i * 5.8)
                add_image_or_placeholder(
                    slide, img_cfg, default_left=def_left, default_top=1.6, default_w=5.5, default_h=4.8
                )

        # --- Table Slide (hyperparameters, metrics, ablation) ---[cite: 3]
        elif stype == "table":
            slide = prs.slides.add_slide(prs.slide_layouts[LAYOUT_CONTENT])
            format_slide_header(slide, item.get("title", ""), item.get("section"))
            for ph in list(slide.placeholders):
                if ph != slide.shapes.title:
                    ph.element.getparent().remove(ph.element)
            if "table_data" in item:
                add_styled_table(
                    slide,
                    item["table_data"],
                    left_in=item.get("left_in", 1.0),
                    top_in=item.get("top_in", 1.8),
                    width_in=item.get("width_in", 11.33),
                )

        # --- Text-Only Slide (sources, references) ---[cite: 1, 4]
        elif stype == "text_only":
            slide = prs.slides.add_slide(prs.slide_layouts[LAYOUT_CONTENT])
            format_slide_header(slide, item.get("title", ""), item.get("section"))
            if len(slide.placeholders) > 1:
                body_ph = slide.placeholders[1]
                body_ph.left = Inches(1.0)
                body_ph.top = Inches(1.5)
                body_ph.width = Inches(11.33)
                body_ph.height = Inches(5.0)
                add_formatted_bullets(body_ph.text_frame, item.get("bullets", []), font_size_pt=13, space_after_pt=10)

        # --- Split Slide (bullets left, image/plot right) ---[cite: 3]
        else:
            slide = prs.slides.add_slide(prs.slide_layouts[LAYOUT_CONTENT])
            format_slide_header(slide, item.get("title", ""), item.get("section"))
            if "bullets" in item and len(slide.placeholders) > 1:
                body_ph = slide.placeholders[1]
                body_ph.left = Inches(0.8)
                body_ph.top = Inches(1.5)
                body_ph.width = Inches(5.2)
                body_ph.height = Inches(5.0)
                add_formatted_bullets(body_ph.text_frame, item["bullets"], font_size_pt=15, space_after_pt=12)

            add_image_or_placeholder(
                slide, item.get("image", {}), default_left=6.3, default_top=1.5, default_w=6.2, default_h=5.0
            )

    # 5. Final sanitation pass across all newly created slides
    sanitize_footers_and_masters(prs, token_map)

    prs.save(output_path)
    print(f"[✓] Presentation generated: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate presentation from PPTX template and YAML spec.")
    parser.add_argument("--template", default="presentation/Vorlage.pptx", type=Path, help="Path to input template PPTX")
    parser.add_argument("--spec", default="presentation/slides_spec.yaml", type=Path, help="Path to YAML spec")
    args = parser.parse_args()

    build_presentation(args.template, args.spec)
