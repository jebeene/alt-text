import io, csv, base64
from typing import List, Tuple
import streamlit as st
from openai import OpenAI
from PIL import Image, UnidentifiedImageError
try:
    import magic  # type: ignore
    HAVE_MAGIC = True
except Exception:
    HAVE_MAGIC = False

MODEL = "gpt-5-mini"
MAX_ALT_CHARS = 125

SYSTEM_PROMPT = f"""
You write HTML alt text per W3C/WAI.
- Concise, <= {MAX_ALT_CHARS} chars
- No "image of" / "picture of"
- Include salient details; include visible on-image text briefly when clear
- Include brand/product if clearly visible (once)
- Return only the alt string
"""

USER_PROMPT = f"Write alt text for this image for an HTML alt attribute, under {MAX_ALT_CHARS} chars."

def sniff_mime(buf: bytes, filename: str | None) -> str:
    if HAVE_MAGIC:
        try:
            m = magic.Magic(mime=True)
            return m.from_buffer(buf)
        except Exception:
            pass
    # fallback by extension
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

def generate_alt(client: OpenAI, file) -> str:
    # file is a Streamlit UploadedFile
    raw = file.getvalue()

    # sanity check that it's an image
    try:
        Image.open(io.BytesIO(raw)).verify()
    except (UnidentifiedImageError, OSError):
        return "(not an image)"

    data_url = to_data_url(raw, file.name)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ],
        temperature=0.2,
    )
    text = (resp.choices[0].message.content or "").strip()
    return text[:MAX_ALT_CHARS].rstrip()

st.set_page_config(page_title="Alt Text Generator", page_icon="🖼️", layout="centered")
st.title("🖼️ Alt Text Generator")

with st.sidebar:
    st.header("🔑 OpenAI API Key")
    api_key = st.text_input("Enter your API key", type="password", help="Key is kept only in this session.")
    st.caption("Tip: create a key in the OpenAI dashboard and paste it here.")
    st.divider()
    st.header("Settings")
    max_chars = st.slider("Max characters", 60, 200, MAX_ALT_CHARS, 5)

if api_key:
    client = OpenAI(api_key=api_key)
    st.session_state.setdefault("rows", [])  # [(name, alt, chars)]

    files = st.file_uploader(
        "Upload images",
        type=["jpg","jpeg","png","webp","gif","bmp","tiff"],
        accept_multiple_files=True
    )

    if files:
        if st.button("Generate alt text for all"):
            rows: List[Tuple[str, str, int]] = []
            for f in files:
                alt = generate_alt(client, f)
                # enforce slider cap
                alt = alt[:max_chars].rstrip()
                rows.append((f.name, alt, len(alt)))
            st.session_state["rows"] = rows

    # Show results
    if st.session_state.get("rows"):
        st.subheader("Results")
        for name, alt, n in st.session_state["rows"]:
            st.image([f for f in files if f.name == name][0], caption=name, width=220)
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
    st.warning("Enter your OpenAI API key in the sidebar to continue.")
