import io, csv, base64
from typing import List, Tuple
from html import escape

import streamlit as st
from openai import OpenAI
from PIL import Image, UnidentifiedImageError

# ---------- Minimalist UI styles (Apple/Airbnb vibe) ----------
st.markdown("""
<style>
:root{
  --card-bg:#ffffff; --border:#e7e7e9; --text:#111113; --muted:#666a73; --pill:#f5f5f7;
  --shadow:0 1px 2px rgba(16,24,40,.06), 0 8px 24px rgba(16,24,40,.08);
}
@media (prefers-color-scheme: dark){
  :root{ --card-bg:#161618; --border:#242428; --text:#ECECEE; --muted:#9b9ca3; --pill:#1f1f23;
         --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.5); }
}
html, body, .stApp { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial; }
.block-container { padding-top: 2.2rem; }

.alt-card{
  display:grid; grid-template-columns:minmax(180px, 320px) 1fr; gap:20px;
  background:var(--card-bg); border:1px solid var(--border);
  border-radius:16px; padding:16px; box-shadow:var(--shadow); align-items:start; margin-bottom:18px;
}
.alt-card img{ width:100%; height:auto; border-radius:12px; border:1px solid var(--border); }
.alt-meta{ color:var(--muted); font-size:0.9rem; margin:4px 0 10px; }
.alt-copy-row{ display:flex; gap:10px; align-items:start; }
.alt-ta{
  flex:1; min-height:92px; max-height:160px; resize:none;
  padding:10px 12px; border-radius:10px; border:1px solid var(--border);
  background:transparent; color:var(--text); line-height:1.45; outline:none;
}
.alt-copy{
  border:1px solid var(--border); background:var(--pill); color:var(--text);
  padding:8px 12px; border-radius:999px; cursor:pointer;
}
.alt-note{ font-size:.85rem; color:#16a34a; min-height:1em; margin-top:6px; opacity:0; transition:opacity .15s ease; }
</style>
""", unsafe_allow_html=True)

MODEL = "gpt-4o-mini"

FORMAT_TO_MIME = {
    "JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp",
    "GIF": "image/gif", "BMP": "image/bmp", "TIFF": "image/tiff",
}

def sniff_mime(buf: bytes, filename: str | None) -> str:
    try:
        img = Image.open(io.BytesIO(buf))
        mime = FORMAT_TO_MIME.get(img.format)
        if mime: return mime
    except Exception:
        pass
    if filename:
        lower = filename.lower()
        if lower.endswith(".png"): return "image/png"
        if lower.endswith(".webp"): return "image/webp"
        if lower.endswith(".gif"): return "image/gif"
        if lower.endswith(".bmp"): return "image/bmp"
        if lower.endswith((".tif",".tiff")): return "image/tiff"
    return "image/jpeg"

def to_data_url(buf: bytes, filename: str | None) -> str:
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:{sniff_mime(buf, filename)};base64,{b64}"

def make_system_prompt(max_chars: int) -> str:
    return (
        "You write HTML alt text per W3C/WAI.\n"
        f"- Concise, <= {max_chars} chars\n"
        '- No "image of" / "picture of"\n'
        "- Include salient details; include visible on-image text briefly when clear\n"
        "- Include brand/product if clearly visible (once)\n"
        "- Return only the alt string"
    )

def make_user_prompt(max_chars: int) -> str:
    return f"Write alt text for this image for an HTML alt attribute, under {max_chars} chars."

def generate_alt(client: OpenAI, file, max_chars: int, creativity: str) -> str:
    temp_map = {"More exact": 0.0, "Balanced (recommended)": 0.2, "More creative": 0.5}
    temperature = temp_map.get(creativity, 0.2)

    raw = file.getvalue()
    try:
        Image.open(io.BytesIO(raw)).verify()
    except (UnidentifiedImageError, OSError):
        return "(not an image)"

    data_url = to_data_url(raw, file.name)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role":"system","content":make_system_prompt(max_chars)},
            {"role":"user","content":[
                {"type":"text","text":make_user_prompt(max_chars)},
                {"type":"image_url","image_url":{"url":data_url}},
            ]},
        ],
        temperature=temperature,
    )
    text = (resp.choices[0].message.content or "").strip()
    return text[:max_chars].rstrip()

# ---------- Card renderer ----------
def render_result_card(file_obj, name: str, alt: str, chars: int, key: str, img_width_px: int = 260, show_count: bool = True):
    raw = file_obj.getvalue()
    img_src = to_data_url(raw, file_obj.name)
    count_bit = f" · {chars} characters" if show_count else ""
    html = f"""
    <div class="alt-card" style="grid-template-columns:{img_width_px}px 1fr;">
      <div>
        <img src="{img_src}" alt="{escape(name)}">
        <div class="alt-meta">{escape(name)}</div>
      </div>
      <div>
        <div class="alt-meta"><strong>Description (read-only)</strong>{count_bit}</div>
        <div class="alt-copy-row">
          <textarea id="ta_{key}" class="alt-ta" readonly>{escape(alt)}</textarea>
          <button class="alt-copy" onclick="
            const ta=document.getElementById('ta_{key}');
            const note=document.getElementById('note_{key}');
            navigator.clipboard.writeText(ta.value).then(()=>{{
              note.textContent='Copied!';
              note.style.opacity='1';
              setTimeout(()=>{{ note.style.opacity='0'; note.textContent=''; }},1500);
            }});
          ">Copy</button>
        </div>
        <div id="note_{key}" class="alt-note"></div>
      </div>
    </div>
    """
    # Height scaled to image width + content; tweak if needed.
    st.components.v1.html(html, height=img_width_px + 190, scrolling=False)

# ---------- UI ----------
st.set_page_config(page_title="Alt Text Generator", page_icon="🖼️", layout="centered")
st.title("🖼️ Alt Text Generator")
st.info("Upload images, then click **Generate descriptions**. Copy any description with one click.")

with st.sidebar:
    st.header("🔑 OpenAI Key")
    api_key = st.text_input("Paste your OpenAI API key", type="password", help="Used only for this session.")
    st.divider()
    st.header("⚙️ Settings")
    max_chars = st.slider("Maximum description length", 60, 200, 125, 5,
                          help="Shorter is snappier; longer allows a bit more detail.")
    creativity = st.radio("Writing style", ["More exact", "Balanced (recommended)", "More creative"], index=1,
                          help="Choose how precise or creative the descriptions should be.")
    img_size = st.radio("Image size", ["Compact", "Comfortable", "Large"], index=1,
                        help="How big images appear in the results.")
    show_count = st.checkbox("Show character count under each description", True)
    csv_name = st.text_input("Download file name", "alt_text.csv")

if api_key:
    client = OpenAI(api_key=api_key)
    st.session_state.setdefault("rows", [])          # [(name, alt, chars)]
    st.session_state.setdefault("files_by_name", {}) # name -> file

    files = st.file_uploader("Upload images",
                             type=["jpg","jpeg","png","webp","gif","bmp","tiff"],
                             accept_multiple_files=True)

    if files:
        st.session_state["files_by_name"] = {f.name: f for f in files}
        if st.button("Generate descriptions", type="primary"):
            rows: List[Tuple[str, str, int]] = []
            for f in files:
                alt = generate_alt(client, f, max_chars, creativity)
                rows.append((f.name, alt, len(alt)))
            st.session_state["rows"] = rows

    # Results (card layout)
    if st.session_state.get("rows"):
        st.subheader("Results")
        width_map = {"Compact": 200, "Comfortable": 260, "Large": 340}
        img_px = width_map[img_size]
        for name, alt, n in st.session_state["rows"]:
            file_obj = st.session_state["files_by_name"].get(name)
            if file_obj:
                render_result_card(file_obj, name, alt, n, key=f"card_{name}", img_width_px=img_px, show_count=show_count)

        # CSV download
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["filename", "alt_text", "chars"])
        for name, alt, n in st.session_state["rows"]:
            writer.writerow([name, alt, n])
        st.download_button("⬇️ Download CSV", data=out.getvalue().encode("utf-8"),
                           file_name=csv_name, mime="text/csv")
else:
    st.warning("Add your OpenAI API key in the sidebar to continue.")
