# Multimodal RAG

Your documents aren't all text. They contain diagrams, screenshots, charts, tables, and
scanned pages. Multimodal RAG means being able to retrieve and answer from those too.

Start with a warning: **most "multimodal RAG" problems are actually parsing problems.**
Before reaching for image embeddings, check whether your PDF parser is destroying tables.
That's usually where the quality went.

---

## What you're actually dealing with

Different content types need different handling. Sorting them out first saves a lot of
wasted work.

| Content | Common approach |
|---|---|
| Text in a PDF | Normal parsing — but check the output |
| Scanned page | OCR, then treat as text |
| Table | Extract structure, keep it as text |
| Chart or graph | Describe it with a vision model |
| Diagram / architecture drawing | Describe it with a vision model |
| Screenshot | Depends — often OCR plus a description |
| Photo | Describe it |
| Slide | Both: extract text and describe the layout |

Notice how much of that column says "turn it into text." That's the main strategy, and
it's usually right.

---

## Strategy 1: describe images as text (start here)

Run a vision model over each image, write a description, index the description as text.

```python
DESCRIBE_PROMPT = """Describe this image so someone can find it by searching
and understand it without seeing it.

Include:
- What kind of image it is (chart, diagram, screenshot, photo, table)
- The main subject and any title or labels
- For charts: the axes, the trend, and any specific values you can read
- For diagrams: the components and how they connect
- Any text visible in the image

Be specific and factual. Do not speculate about what isn't shown."""


async def describe_image(image_bytes, vision_model, surrounding_text=""):
    prompt = DESCRIBE_PROMPT
    if surrounding_text:
        # Context from the page helps a lot — the caption usually says
        # what the figure is for
        prompt += f"\n\nText near this image:\n{surrounding_text[:500]}"

    return await vision_model(image=image_bytes, prompt=prompt, temperature=0.0)
```

Then index it like any other chunk:

```python
@dataclass
class ImageChunk:
    id: str
    doc_id: str
    page: int
    description: str          # what you embed and search
    image_ref: str            # where the actual image lives
    surrounding_text: str
```

**Why this works well:**

- One search index for everything. No fusing across modalities.
- All your existing machinery applies — hybrid search, reranking, filtering.
- Descriptions contain words, so keyword search finds them too.
- Cheap at query time. The expensive part happens once, at indexing.
- Easy to debug — you can read what the system thinks the image says.

**The limits:**

- The description is lossy. Anything not described is unfindable.
- It costs a vision call per image at indexing.
- Quality depends entirely on the prompt and the model.

For most systems this is the right answer, and the more complex approaches should have to
prove they're better.

---

## Strategy 2: embed images directly

Use a model that puts images and text in the same vector space, so you can search images
with a text query directly.

```python
def index_image(image, model):
    return model.embed_image(image)      # same space as text embeddings

def search_images(query, model, index):
    return index.search(model.embed_text(query))
```

**Good for:** photo libraries, product catalogs, "find me a picture of X." Visual
similarity is the thing you care about.

**Weaker for:** documents. Joint image-text embedding models are typically trained on
photo captions, not on technical diagrams or charts. They'll find "a picture with a graph
in it" but struggle with "the chart showing Q3 revenue declining in EMEA."

For document RAG, descriptions usually beat direct image embedding. Test both if it
matters.

---

## Strategy 3: keep the image for answering

Retrieve using a text description, then send the **actual image** to the model when
generating the answer.

```python
async def answer_with_images(question, model):
    hits = retrieve(question, k=5)

    content = [{"type": "text", "text": f"Question: {question}\n\nContext:"}]

    for hit in hits:
        if hit.is_image:
            content.append({"type": "text", "text": f"[{hit.id}] {hit.description}"})
            content.append({"type": "image", "source": load_image(hit.image_ref)})
        else:
            content.append({"type": "text", "text": f"[{hit.id}] {hit.text}"})

    return await model(content)
```

This is the best-quality option, because the model can read values off the chart itself
rather than relying on your description having captured them.

**Costs:** images are expensive in tokens, and slower. Use it selectively — only send the
image when the question seems to need visual detail:

```python
VISUAL_WORDS = {"chart", "graph", "diagram", "figure", "image", "screenshot",
                "show", "look", "picture", "trend", "axis"}

def needs_actual_image(question, hit):
    return hit.is_image and bool(VISUAL_WORDS & set(question.lower().split()))
```

---

## Tables deserve their own treatment

Tables are the most common multimodal content and the most commonly mangled.

**The problem:** naive PDF text extraction turns a table into a jumble. Columns
interleave, headers detach from data, and the meaning is gone.

```
Product    Q1    Q2       →     Product Q1 Q2 Widget 400 520 Gadget 310 290
Widget     400   520
Gadget     310   290
```

That second version is nearly useless — nothing tells you 520 is Widget's Q2.

**Fixes, in order:**

**1. Use a parser that understands tables.** This is the highest-value fix and it's a
tooling choice, not an AI one. Check what your parser produces before building anything
on top of it.

**2. Convert to a structured text format.** Markdown tables preserve relationships and
models read them well:

```python
def table_to_markdown(rows):
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in body:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)
```

**3. Add a description.** A sentence saying what the table shows helps retrieval a lot,
because the table's own cells are mostly numbers:

```python
async def describe_table(table_md, caption, model):
    return await model(f"""
In one or two sentences, say what this table shows — the subject, the
columns, and the time period or scope if visible.

Caption: {caption}
Table:
{table_md}
""", temperature=0.0)
```

**4. Never split a table across chunks without repeating the header.** Half a table with
no headers has no meaning.

---

## Scanned documents and OCR

If pages are images of text, you need OCR before anything else works.

Practical points:

**OCR quality varies enormously** with scan quality. A clean 300-dpi scan is nearly
perfect; a fax of a fax is not.

**Keep the confidence scores.** Low-confidence OCR is worth flagging rather than silently
indexing as fact:

```python
def index_ocr_page(page, ocr_result, threshold=0.7):
    if ocr_result.mean_confidence < threshold:
        # Low quality — flag it so answers from it can be caveated,
        # and so someone can review these pages
        page.metadata["ocr_low_confidence"] = True
    return page
```

**Keep page images.** For low-confidence pages, sending the actual image to a vision
model at answer time beats trusting bad OCR text.

**A vision model can be an OCR alternative.** For difficult documents — handwriting,
unusual layouts — asking a vision model to transcribe often beats traditional OCR. It
costs more per page, so route by difficulty rather than using it everywhere.

---

## Slides

Presentation slides are a mixed case worth calling out: they have text, layout meaning,
and often a chart that carries the actual message.

Best handling: **one chunk per slide**, combining everything.

```python
async def index_slide(slide, vision_model):
    parts = []

    if slide.title:
        parts.append(f"Slide {slide.number}: {slide.title}")
    if slide.body_text:
        parts.append(slide.body_text)
    if slide.speaker_notes:
        parts.append(f"Notes: {slide.speaker_notes}")     # often the real content
    if slide.has_visual:
        parts.append(await describe_image(slide.image, vision_model))

    return Chunk(
        id=f"{slide.deck_id}#slide{slide.number}",
        text="\n\n".join(parts),
        metadata={"deck": slide.deck_id, "slide": slide.number},
    )
```

Speaker notes are the underrated part. They frequently contain the explanation that the
slide itself only gestures at.

---

## Cost

Indexing images is the expensive part. Do the arithmetic before committing:

```python
def indexing_cost(n_images, tokens_per_image, output_tokens,
                  price_in, price_out):
    """Vision calls at index time."""
    cost_in = n_images * tokens_per_image / 1e6 * price_in
    cost_out = n_images * output_tokens / 1e6 * price_out
    return cost_in + cost_out

# 100,000 images, ~1,500 tokens each, ~150 tokens of description
# at $1/M in and $3/M out → $150 + $45 = $195. Fine.
#
# 10,000,000 images → $19,500. Now it needs a decision.
```

Ways to reduce it:

- **Skip decorative images.** Logos, dividers, and stock photos aren't worth describing.
  Filter by size and position, or by a cheap classifier.
- **Deduplicate.** The same logo appears on every page. Hash the image bytes.
- **Use a smaller vision model** for simple images, a larger one for complex charts.
- **Describe on demand** for rarely-accessed documents, rather than everything up front.

```python
def worth_describing(image, page):
    if image.width < 100 or image.height < 100:
        return False                          # icon or bullet
    if image.bytes_hash in KNOWN_DECORATIVE:  # logos, headers
        return False
    if image.area / page.area < 0.02:
        return False                          # tiny
    return True
```

---

## Things that go wrong

**Parsing was the real problem.** Worth repeating, because it's the most common case. If
tables are being destroyed at parse time, no retrieval strategy fixes it. Look at your
parsed text.

**Descriptions miss the specific values.** "A bar chart showing quarterly revenue" is
findable but can't answer "what was Q3 revenue?" If numbers matter, ask for them
explicitly in the description prompt.

**Images lose their page context.** A chart on page 40 of a report needs to be associated
with the section it's in. Include surrounding text and headings, same as
[contextual retrieval](contextual-retrieval.md).

**Wrong descriptions, indexed silently.** The vision model misreads a chart and you've
embedded a wrong description. Sample and check. Retrieval quietly degrades otherwise.

**Citations point at the description, not the image.** Users need to see the actual
figure. Store the image reference and show it.

**Everything is treated as an image.** A text-based PDF doesn't need vision — you're
paying for OCR on text you could have extracted directly. Detect which is which.

---

## Interview questions

**1. "Your documents have charts and tables. How do you handle them?"**

Start by checking parsing — most of these problems are parsing problems. Then: describe
images with a vision model at index time and search the descriptions; convert tables to
markdown and add a description; optionally send the real image at answer time for visual
questions.

**2. "Why not embed the images directly?"**

Joint image-text embedding models are mostly trained on photo captions, not technical
diagrams. They find "a picture with a graph" but not "the chart showing Q3 decline in
EMEA." Descriptions usually work better for documents. Direct embedding is better for
photo libraries.

**3. "Table data gets mangled. What do you do?"**

Fix parsing first — use a table-aware parser. Convert to markdown to keep the
relationships. Add a description because the cells are mostly numbers and don't retrieve
well. Never split without repeating headers.

**4. "Describing 10 million images costs $20k. Options?"**

Skip decorative images by size and position, deduplicate by hash, use a smaller model for
simple images, and describe rarely-accessed documents on demand rather than up front.

**5. "How do you handle scanned documents?"**

OCR, keep confidence scores, flag low-confidence pages, and keep page images so you can
fall back to a vision model for the bad ones. Route difficult documents to a vision model
instead of traditional OCR.

**6. "How do you evaluate multimodal retrieval?"**

Same as text, but segmented by content type — text, table, chart, diagram. The failure
modes differ, and an overall number hides which type is broken.

---

## What to remember

- Check your parsing first. Most multimodal problems are parsing problems.
- Turning images into text descriptions is the main strategy, and usually the right one.
- Direct image embedding suits photos, not technical documents.
- Send the real image at answer time only for visually-specific questions.
- Tables: table-aware parser, markdown format, add a description, repeat headers.
- Include surrounding text when describing an image — captions carry the meaning.
- Ask for specific values in the description if numbers matter.
- Skip decorative images and deduplicate to control cost.
- Store image references so citations can show the actual figure.
- Evaluate segmented by content type.

---

**Next:** [rag-debugging.md](rag-debugging.md) — finding out why the answer was wrong.
