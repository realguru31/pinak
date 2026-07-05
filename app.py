"""
NSE v3 response diagnostic — run this INSTEAD of app.py to see exactly what the
Streamlit environment gets back from NSE.

    streamlit run diagnostic.py

It fetches your known-good URL and prints status, Content-Type, Content-Encoding,
raw bytes, and whether JSON parses — so we can tell a real IP block apart from a
brotli/encoding decode problem (the usual cause of
"Expecting value: line 1 column 1 (char 0)").
"""

import requests
import streamlit as st

st.set_page_config(page_title="NSE v3 diagnostic", layout="wide")
st.title("NSE option-chain-v3 diagnostic")

URL = ("https://www.nseindia.com/api/option-chain-v3"
       "?type=Indices&symbol=NIFTY&expiry=07-July-2026")

BASE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain",
    "X-Requested-With": "XMLHttpRequest",
}

st.write("**Target URL**")
st.code(URL)

enc = st.radio(
    "Accept-Encoding header to send",
    ["gzip, deflate  (safe — no brotli needed)",
     "gzip, deflate, br  (current app.py setting)",
     "(send nothing)"],
    index=0,
)

# report whether brotli decoders are importable in this environment
brotli_available = False
for mod in ("brotli", "brotlicffi"):
    try:
        __import__(mod)
        brotli_available = True
        st.info(f"`{mod}` IS importable in this environment.")
        break
    except Exception:
        pass
if not brotli_available:
    st.warning("No brotli decoder (`brotli`/`brotlicffi`) importable — if NSE "
               "replies with Content-Encoding: br, requests cannot decode it and "
               "`.json()` will fail on byte 1.")

if st.button("Run diagnostic", type="primary"):
    headers = dict(BASE_HEADERS)
    if enc.startswith("gzip, deflate,"):
        headers["Accept-Encoding"] = "gzip, deflate, br"
    elif enc.startswith("gzip, deflate "):
        headers["Accept-Encoding"] = "gzip, deflate"
    # "(send nothing)" -> omit the header entirely

    s = requests.Session()
    s.headers.update(headers)

    warm = []
    try:
        for u in ("https://www.nseindia.com", "https://www.nseindia.com/option-chain"):
            w = s.get(u, timeout=15)
            warm.append((u, w.status_code, w.headers.get("Content-Encoding")))
    except Exception as e:
        st.error(f"Warm-up request failed: {e}")

    st.subheader("Warm-up")
    st.write(warm)
    st.write("Cookies obtained:", dict(s.cookies.get_dict()))

    try:
        r = s.get(URL, timeout=20)
    except Exception as e:
        st.error(f"API request raised: {e}")
        st.stop()

    st.subheader("API response")
    c1, c2, c3 = st.columns(3)
    c1.metric("HTTP status", r.status_code)
    c2.metric("Content-Encoding", r.headers.get("Content-Encoding") or "(none)")
    c3.metric("Bytes", len(r.content))
    st.write("**Content-Type:**", r.headers.get("Content-Type"))
    st.write("**Request Accept-Encoding sent:**", headers.get("Accept-Encoding", "(none)"))

    st.write("**First 300 raw bytes (repr):**")
    st.code(repr(r.content[:300]))
    st.write("**First 500 chars of r.text:**")
    st.code(r.text[:500] if r.text else "(empty)")

    try:
        d = r.json()
        st.success("JSON parsed OK ✓")
        rec = d.get("records", {})
        st.write("records keys:", list(rec.keys()))
        st.write("underlyingValue:", rec.get("underlyingValue"))
        st.write("expiryDates:", rec.get("expiryDates"))
        st.write("row count:", len(rec.get("data", [])))
    except Exception as e:
        st.error(f"r.json() failed: {e}")
        # if it looks like brotli, try to decode manually to confirm the theory
        if (r.headers.get("Content-Encoding") == "br") and not brotli_available:
            st.warning("Confirmed: server sent brotli and no decoder is installed. "
                       "Fix = remove `br` from Accept-Encoding, or add `brotli` to "
                       "requirements.txt.")
