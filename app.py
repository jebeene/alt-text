# async_alt_text_app.py
import io, csv, base64, asyncio
from typing import List, Tuple
import streamlit as st
from openai import AsyncOpenAI
from PIL import Image, UnidentifiedImageError

try:
    import magic  # type: ignore
    HAVE_MAGIC = True
except Exception:
    HAVE_MAGIC = False

MODEL = "gpt-5-mini"
DEFAULT_MAX_ALT_CHARS = 125
CONCURRENCY = 5  # run 5 images at a time

def sniff_mime(buf: bytes, filename: str | None) -> str:
    if HAVE_MAGIC:
        try:
            m = magic.Magic(mime=True)
            return m.from_buffer(buf)
        except Exception:
            pass
    if filename and filename.lower().endswith((".png",)):
        return "image/png"
    if filename and filename.lower().endswith((".webp",)):
        return "image/webp"
    if filename and filename.lower().endswith((".gif",)):
        return "image/gif"
    return "image/jpeg"

def to_data_url(buf: bytes, filename: str | None) -> str:
    mime = sniff_mime(buf, filename)
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def build_prompts(max_chars: int) -> tuple[str, str]:
    system_prompt = (
        "You write HTML alt text per W3C/WAI.\n"
        f"- Concise, <= {max_chars} chars\n"
        '- No "image of" / "picture of"\n'
        "- Include salient details; include visible on-image text briefly when clear\n"
        "- Include brand/product if clearly visible (once)\n"
        "- Return only the alt string"
    )
    user_prompt = f"Write alt text for this image for an HTML alt attribute, under {max_chars} chars."
    return system_prompt, user_prompt

def verify_image_or_placeholder(file) -> tuple[bytes, bool]:
    raw = file.getvalue()
    try:
        Image.open(io.BytesIO(raw)).verify()
        return raw, True
    except (UnidentifiedImageError, OSError):
        return raw, False

async def generate_alt_async(client: AsyncOpenAI, file, max_chars: int, sem: asyncio.Semaphore) -> tuple[str, str, int]:
    """
    Returns (filename, alt_text, char_count). If not an image, alt_text is "(not an image)".
    """
    async with sem:
        raw, ok = verify_image_or_placeholder(file)
        if not ok:
            return file.name, "(not an image)", len("(not an image)")

        data_url = to_data_url(raw, file.name)
        system_prompt, user_prompt = build_prompts(max_chars)

        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        text = text[:max_chars].rstrip()
        return file.name, text, len(text)

async def process_all(files, api_key: str, max_chars: int) -> List[Tuple[str, str, int]]:
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [generate_alt_async(client, f, max_chars, sem) for f in files]
    results = []
    # optional: simple progress indicator
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)
    # preserve original file order in UI
    order = {f.name: i for i, f in enumerate(files)}
    results.sort(key=lambda r: order.get(r[0], 0))
    return results

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Alt Text Generator", page_icon="🖼️", layout="centered")
st.title("🖼️ Alt Text Generator")

with st.sidebar:
    st.header("🔑 OpenAI API Key")
    api_key = st.text_input("Enter your API key", type="password", help="Key is kept only in this session.")
    st.caption("Tip: create a key in the OpenAI dashboard and paste it here.")
    st.divider()
    st.header("Settings")
    max_chars = st.slider("Max characters", 60, 200, DEFAULT_MAX_ALT_CHARS, 5)
    st.caption(f"Concurrency: {CONCURRENCY}")

if api_key:
    st.session_state.setdefault("rows", [])  # [(name, alt, chars)]
    st.session_state.setdefault("file_bytes", {})  # cache image bytes for display

    files = st.file_uploader(
        "Upload images",
        type=["jpg","jpeg","png","webp","gif","bmp","tiff"],
        accept_multiple_files=True
    )

    if files:
        # cache bytes to avoid re-reading after async pass
        for f in files:
            st.session_state["file_bytes"][f.name] = f.getvalue()

        if st.button("Generate alt text for all"):
            with st.spinner("Generating alt text…"):
                # Run the async batch with bounded concurrency
                rows: List[Tuple[str, str, int]] = asyncio.run(process_all(files, api_key, max_chars))
                st.session_state["rows"] = rows

    # Show results
    if st.session_state.get("rows"):
        st.subheader("Results")
        # Recreate lightweight UploadedFile-like objects for display from cached bytes
        for name, alt, n in st.session_state["rows"]:
            buf = st.session_state["file_bytes"].get(name)
            if buf:
                st.image(buf, caption=name, width=220)
            else:
                st.write(f"*(preview unavailable for {name})*")
            st.text_area(f"Alt for {name}", alt, height=60, key=f"alt_{name}")
            st.caption(f"{n} characters")

        # Download CSV
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["filename", "alt_text", "chars"])
        for name, alt, n in st.session_state["rows"]:
            writer.writerow([name, alt, n])
        st.download_button(
            "⬇️ Download CSV",
            data=out.getvalue().encode("utf-8"),
            file_name="alt_text.csv",
            mime="text/csv",
        )
else:
    st.info("Add your OpenAI API key in the sidebar to begin.")
