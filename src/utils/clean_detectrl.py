# -*- coding: utf-8 -*-
"""
DetectRL Merged Cleaning Pipeline (Stages 1-7)
Combines functionality of clean_detectrl_dir1 through clean_detectrl_dir7.
Executes cleaning passes sequentially in memory for maximum efficiency.

Updates:
- Added Stage 7: Conservative confusable/homoglyph fixing and invisible character removal.
"""

from __future__ import annotations
import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

# ==============================================================================
# SHARED UTILITIES
# ==============================================================================

def detect_format(path: Path) -> str:
    return "jsonl" if path.suffix.lower() == ".jsonl" else "json"

def read_records_raw(path: Path) -> Tuple[Any, List[Dict[str, Any]]]:
    """Reads records, preserving the outer wrapper if it exists (for JSON)."""
    fmt = detect_format(path)
    if fmt == "jsonl":
        recs = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        return None, recs
    else:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return raw, raw
        if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
            return raw, raw["data"]
        if isinstance(raw, dict):
            return raw, [raw]
        return None, []

def write_records(path: Path, raw_obj: Any, recs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = detect_format(path)
    if fmt == "jsonl":
        with path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        out = recs
        if isinstance(raw_obj, dict) and "data" in raw_obj and isinstance(raw_obj["data"], list):
            raw_obj["data"] = recs
            out = raw_obj
        elif isinstance(raw_obj, dict) and recs and raw_obj is recs[0]:
            out = recs[0]
        with path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

# ==============================================================================
# STAGE 1: clean_detectrl_dir1.py (The "Gold" Version)
# ==============================================================================
class Stage1:
    REFUSAL_REGEXES = [
        re.compile(r"(?:i|we)\W*(?:m|am|are)?\s+not\s+(?:able|bale|abel|albe|abli)\s+to\s+help", re.IGNORECASE),
        re.compile(r"(?:i|we)\s+cannot\s+(?:fulfill|fulfil|comply|produce|generate|write)", re.IGNORECASE),
        re.compile(r"as\s+(?:an?|the)\s+(?:ai|artificial\s+intelligence|large\s+language|language)\s+model", re.IGNORECASE),
        re.compile(r"only\s+a\s+lan[guage]{3,6}\s+model", re.IGNORECASE),
        re.compile(r"anthropic'?s?\s+ai", re.IGNORECASE),
        re.compile(r"usage\s+guidelines", re.IGNORECASE),
        re.compile(r"violate\s+(?:safety|ethical|community)\s+protocols", re.IGNORECASE),
        re.compile(r"cannot\s+generate\s+(?:inappropriate|sexual|violent|harmful)", re.IGNORECASE),
        re.compile(r"i\s+am\s+unable\s+to", re.IGNORECASE),
        re.compile(r"regrettably,\s+as\s+a", re.IGNORECASE),
    ]
    RE_GLOBAL_TAGS = [
        re.compile(r"\[\s*(?:system|assistant|user|model)\s*\]", re.IGNORECASE),
        re.compile(r"\[\s*(?:s[yij]+s.*?|a[s$]+i[s$]+.*?|u[sz]er.*?)\s*\]", re.IGNORECASE),
        re.compile(r"(?:^|\n)\s*(?:System|Assistant|User|Model)\s*:", re.IGNORECASE),
    ]
    RE_START_ARTIFACTS = [
        re.compile(r"^\s*Sure\s*[,!.]", re.IGNORECASE),
        re.compile(r"^\s*Here(?:\s+(?:is|are)|\'s)\s+(?:the|a|an)\s+", re.IGNORECASE),
        re.compile(r"^\s*Okay\s*[,!.]", re.IGNORECASE),
        re.compile(r"^\s*Certainly\s*[,!.]", re.IGNORECASE),
        re.compile(r"^\s*Of\s+course\s*[,!.]", re.IGNORECASE),
    ]
    RE_NN_START = re.compile(r"^\s*nn(?=[A-Z])")
    RE_NN_INLINE = re.compile(r"(?<=[a-z.,!?])\s*nn(?=[A-Z])")
    RE_ABSTRACT_STICKY = re.compile(r"abstract(?=[A-Z])", re.IGNORECASE)
    RE_WRAPPER_ANCHOR = re.compile(
        r"(?<![A-Za-z])(?:here(?:\s+(?:is|are|s)|\'s)|today\s+is|sure[,!]?\s*here(?:\s+(?:is|are|s)|\'s))",
        re.IGNORECASE,
    )
    WRAPPER_KEYWORDS_EXTENDED = [
        "news", "article", "headline", "body", "abstract", "review", "story",
        "sentence", "sentences", "based on", "continuation", "summary", "prompt",
        "requested", "possible", "polished", "version", "attempt",
        "experience", "opinion", "take", "thoughts", "list", "pros", "cons",
        "overview", "analysis", "essay", "blog", "post", "comment", "draft",
        "example", "revised", "text", "rewrite", "writing", "copy", "below"
    ]
    SENT_END_CHARS = set(["?", "!", ".", ":"])

    @staticmethod
    def remove_span_to_terminator(text: str, start_idx: int) -> str:
        if start_idx < 0 or start_idx >= len(text): return text
        end = len(text) - 1
        for i in range(start_idx, len(text)):
            if text[i] in Stage1.SENT_END_CHARS or text[i] == '\n':
                end = i
                break
        colon_pos = text.find(":", start_idx, min(len(text), start_idx + 300))
        if colon_pos != -1 and colon_pos < end: end = colon_pos
        new_text = (text[:start_idx] + text[end + 1:]).strip()
        new_text = re.sub(r'^[.!?,:;]+\s*', '', new_text)
        return new_text.strip()

    @staticmethod
    def clean(text: str) -> Optional[str]:
        if not isinstance(text, str) or not text.strip():
            return text

        s = text
        s = Stage1.RE_NN_START.sub("", s)
        s = nfkc(s).strip()
        s = Stage1.RE_NN_START.sub("", s)
        s = Stage1.RE_NN_INLINE.sub(" ", s)

        for regex in Stage1.REFUSAL_REGEXES:
            if regex.search(s):
                return None

        for _ in range(5):
            before_iter = s
            for r in Stage1.RE_GLOBAL_TAGS:
                s = r.sub(" ", s).strip()
            if s.startswith(":"):
                s = s[1:].strip()
            s = re.sub(r"\s+", " ", s)
            s = Stage1.RE_ABSTRACT_STICKY.sub("Abstract ", s).strip()
            for r in Stage1.RE_START_ARTIFACTS:
                 match = r.match(s)
                 if match:
                     if "here" not in match.group(0).lower():
                         s = r.sub("", s).strip()
            m = Stage1.RE_WRAPPER_ANCHOR.search(s)
            if m:
                start = m.start()
                window = s[start:min(len(s), start + 200)]
                txt = window.lower()
                looks_like_wrapper = any(k in txt for k in Stage1.WRAPPER_KEYWORDS_EXTENDED)
                if start < 200 and looks_like_wrapper:
                     s = Stage1.remove_span_to_terminator(s, start)

            if s == before_iter:
                break

        if not s.strip(): return None
        return s

# ==============================================================================
# STAGE 2: clean_detectrl_dir2.py (The "Black Hole")
# ==============================================================================
class Stage2:
    RE_ORDINAL_LABEL_V11 = re.compile(
        r"(?:\b(?:"
            r"based|on|the|a|an|given|provided|generated|requested|review['’]?s?|story['’]?s?|sentence|of"
        r")\s+)*"
        r"(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|Last|\d+(?:st|nd|rd|th))"
        r"(?:\s+and\s+the\s+Last)?"
        r"\s+Sentence"
        r"\s*[:\.\-]+\s*",
        re.IGNORECASE
    )
    RE_BRUTE_FORCE_HEADER = re.compile(r"\b(?:Title|Headline|Abstract|Subject|Topic)\s*:\s*", re.IGNORECASE)
    RE_SENTENCE_COUNT_V11 = re.compile(
        r"(?:\b(?:with|of|output|write|generate|only|full|total|summary|limit)\s+)?"
        r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty)\s+"
        r"sentences?"
        r"(?:\s*[:\.\-\(\)])",
        re.IGNORECASE
    )
    RE_INSTRUCTION_LEAK = re.compile(
        r"(?:\s|^)Here['’]s\s+a\s+(?:polished|rewritten|improved|shortened)\s+version.*?(?:[:\.]|$)",
        re.IGNORECASE | re.DOTALL
    )
    NUM_PATTERN = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|short|brief|few)"
    RE_META_HEADER_START = re.compile(
        r"^.*?"
        r"(?:\b(?:is|a|an|the|write|provide)\b\s+)*"
        f"{NUM_PATTERN}\s*"
        r"[-]?\s*sentences?\s+"
        r"(?:article|summary|review|story|abstract|continuation|narrative)?"
        r".*?"
        r"(?:[:\n]|\.(?=\s)|$)",
        re.IGNORECASE | re.DOTALL
    )
    REFUSAL_REGEXES = [
        re.compile(r"(?:i|we)\W*(?:m|am|are)?\s+not\s+(?:able|bale)\s+to\s+help", re.IGNORECASE),
        re.compile(r"(?:i|we)\s+cannot\s+(?:fulfill|fulfil|comply|produce|generate)", re.IGNORECASE),
        re.compile(r"as\s+(?:an?|the)\s+(?:ai|artificial\s+intelligence|language)\s+model", re.IGNORECASE),
        re.compile(r"only\s+a\s+language\s+model", re.IGNORECASE),
        re.compile(r"ethics\s+guidelines", re.IGNORECASE),
    ]
    RE_GLOBAL_TAGS = [
        re.compile(r"\[\s*(?:system|assistant|user|model|human|llm)\s*\]", re.IGNORECASE),
        re.compile(r"(?:^|\n)\s*(?:System|Assistant|User|Model)\s*:\s*", re.IGNORECASE),
        re.compile(r"^\s*nn(?=[A-Z])"),
    ]

    @staticmethod
    def normalize_punctuation(text: str) -> str:
        if not text: return text
        trans_table = str.maketrans({
            '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'", '\u0060': "'", '\u00b4': "'",
        })
        return text.translate(trans_table)

    @staticmethod
    def clean_pass(s: str) -> str:
        s = Stage2.RE_BRUTE_FORCE_HEADER.sub(" ", s)
        if "version" in s or "Here" in s:
            s = Stage2.RE_INSTRUCTION_LEAK.sub("", s)
        if "sentence" in s.lower():
            s = Stage2.RE_SENTENCE_COUNT_V11.sub(" ", s)
        if "Sentence" in s:
            s = Stage2.RE_ORDINAL_LABEL_V11.sub(" ", s)
        header_window = s[:300]
        match_meta = Stage2.RE_META_HEADER_START.match(header_window)
        if match_meta:
            cut_point = match_meta.end()
            s = (header_window[cut_point:] + s[300:]).strip()
        for r in Stage2.RE_GLOBAL_TAGS:
            s = r.sub(" ", s)
        return s.strip()

    @staticmethod
    def clean(text: str) -> Optional[str]:
        if not isinstance(text, str) or not text.strip():
            return text
        s = nfkc(text)
        s = Stage2.normalize_punctuation(s)
        s = s.strip()

        for regex in Stage2.REFUSAL_REGEXES:
            if regex.search(s): return None

        for _ in range(3):
            prev_s = s
            s = Stage2.clean_pass(s)
            s = re.sub(r"\s+", " ", s).strip()
            if s == prev_s:
                break

        for regex in Stage2.REFUSAL_REGEXES:
            if regex.search(s): return None

        if not s: return None
        return s

# ==============================================================================
# STAGE 3: clean_detectrl_dir3.py (Tail Cutter)
# ==============================================================================
class Stage3:
    TAIL_WINDOW_CHARS = 1200
    MIN_PREFIX_CHARS = 200
    MAX_CUT_FRACTION = 0.45
    RE_TAIL_ANCHOR = re.compile(
        r"(?is)"
        r"("
        r"(?:^|[\n\r]|[.!?]\s+)\s*(?:---+\s*)?(?:editor'?s|author'?s|assistant'?s|side|special)?\s*note\s*:"
        r"|(?:^|[\n\r]|[.!?]\s+)\s*(?:---+\s*)?p\.?\s*s\.?\s*:"
        r"|(?:^|[\n\r]|[.!?]\s+)\s*disclaimer\s*:"
        r"|(?:^|[\n\r]|[.!?]\s+)\s*year\s*:\s*\d{4}"
        r"|(?:^|[\n\r]|[.!?]\s+)\s*date\s*:\s*[A-Za-z]{3,}"
        r"|(?:^|[\n\r]|[.!?]\s+)\s*(?:final\s+)?rating\s*:\s*\d"
        r")"
    )
    META_KEYWORDS = [
        "polished", "rewritten", "improved", "shortened",
        "grammar", "spelling", "clarity", "concision", "readability",
        "minor changes", "made some changes", "i made some", "i have made",
        "i have replaced", "i have added", "i have corrected", "i have changed",
        "can be rephrased", "rephrased as",
        "formatting", "blank lines",
        "maintaining the same meaning", "maintaining the same message",
        "more formal", "news article", "academic style", "tone",
        "original story", "original narration", "can be found on", "the link to the original",
        "reddit.com", "r/writingprompts", "/r/", "u/", "/u/",
    ]
    RE_META = re.compile(r"(?is)\b(" + "|".join(re.escape(k) for k in META_KEYWORDS) + r")\b")
    RE_BULLETS = re.compile(r"(?is)(?:^|\n)\s*(?:[*\-]\s+)(?:i\s+have\b|\".+?\"\s+can\s+be\s+rephrased)")
    RE_ATTRIB = re.compile(r"(?is)(reddit\.com|r/writingprompts|\b/u/\w+|\bu/\w{3,}|(?:^|\s)/r/\w+)")
    RE_SEP_TAIL = re.compile(r"(?is)(?:\s*[-_*]{3,}\s*)+$")

    @staticmethod
    def looks_like_meta_block(post: str) -> bool:
        p = post.strip()
        if not p: return False
        if Stage3.RE_ATTRIB.search(p): return True
        if Stage3.RE_BULLETS.search(p): return True
        if Stage3.RE_META.search(p): return True
        return False

    @staticmethod
    def clean(text: str) -> Optional[str]:
        if not text: return text
        t = text
        for _ in range(3):
            if len(t) < Stage3.MIN_PREFIX_CHARS: break
            tail = t[-Stage3.TAIL_WINDOW_CHARS:] if len(t) > Stage3.TAIL_WINDOW_CHARS else t
            matches = list(Stage3.RE_TAIL_ANCHOR.finditer(tail))
            if not matches: break
            m = matches[-1]
            global_cut_pos = len(t) - len(tail) + m.start()
            if global_cut_pos < Stage3.MIN_PREFIX_CHARS: break
            post = t[global_cut_pos:]
            pre = t[:global_cut_pos]
            if not Stage3.looks_like_meta_block(post): break
            cut_fraction = len(post) / max(1, len(t))
            if cut_fraction > Stage3.MAX_CUT_FRACTION and not Stage3.RE_ATTRIB.search(post): break
            new_t = pre.rstrip()
            new_t = Stage3.RE_SEP_TAIL.sub("", new_t).rstrip()
            new_t = re.sub(r"[\s\-_*]+$", "", new_t).rstrip()
            if new_t == t: break
            t = new_t
        return t

# ==============================================================================
# STAGE 4: clean_detectrl_dir4.py (Refusal + Markdown + Tail V13)
# ==============================================================================
class Stage4:
    LEAD_WINDOW = 280
    TAIL_WINDOW = 1400
    RE_MD_BOLD = re.compile(r"(?s)\*\*(.+?)\*\*")
    RE_MD_BOLD_U = re.compile(r"(?s)__(.+?)__")
    RE_MD_ITALIC = re.compile(r"(?s)(?<!\n)\*(?!\s)(.+?)(?<!\s)\*(?!\w)")
    RE_MD_BARE_STARS = re.compile(r"\*{2,}")
    RE_MD_BARE_UNDERS = re.compile(r"_{2,}")
    RE_TAIL_ANCHOR = re.compile(
        r"(?is)"
        r"("
        r"(?:^|[\n\r]|[.!?]\s+)\s*(?:---+\s*)?"
        r"(?:editor'?s|author'?s|assistant'?s|system)?\s*note\s*:"
        r"|(?:^|[\n\r]|[.!?]\s+)\s*(?:---+\s*)?note\s*:"
        r"|(?:^|[\n\r]|[.!?]\s+)\s*(?:---+\s*)?(?:disclaimer|important\s+note)\s*:"
        r"|(?:^|[\n\r]|[.!?]\s+)\s*(?:---+\s*)?(?:p\.?\s*s\.?|ps)\s*:"
        r"|(?:^|[\n\r]|[.!?]\s+)\s*(?:final\s+)?rating\s*:\s*\d"
        r"|(?:^|[\n\r]|[.!?]\s+)\s*year\s*:\s*\d{4}"
        r"|(?:^|[\n\r]|[.!?]\s+)\s*date\s*:\s*[A-Za-z]{3,}"
        r")"
    )
    RE_ATTRIB = re.compile(r"(?is)(reddit\.com|r/writingprompts|\b/u/\w+|\bu/\w{3,}|(?:^|\s)/r/\w+)")
    RE_BULLETS = re.compile(r"(?is)(?:^|\n)\s*(?:[*\-]\s+)")
    RE_META_TAIL = re.compile(
        r"(?is)\b("
        r"polished|rewritten|improved|grammar|spelling|clarity|concision|readability|"
        r"minor\s+changes?|made\s+some\s+changes?|i\s+made\s+some|i\s+made\s+some|"
        r"rephrased|formatting|blank\s+lines|"
        r"original\s+story|can\s+be\s+found\s+on|the\s+link\s+to\s+the\s+original|"
        r"this\s+(?:review|story|article)\s+(?:has\s+been|was)\s+(?:polished|rewritten|improved)"
        r")\b"
    )
    RE_LEAD_APOLOGY = re.compile(r"(?is)^\s*(i\s+apologize|i'?m\s+sorry|sorry)\b[\s,]*")
    RE_REFUSAL_CUE = re.compile(
        r"(?is)\b("
        r"cannot|can'?t|won'?t|not\s+able|do\s+not\s+feel\s+comfortable|"
        r"not\s+appropriate|ethical|policy|guidelines|"
        r"as\s+an?\s+ai|language\s+model|"
        r"i\s+regret\s+to\s+inform\s+you|i\s+cannot\s+assist|"
        r"against\s+my\s+(?:use\s+case|safety|content)\s+policy|"
        r"i\s+don'?t\s+feel\s+comfortable"
        r")\b"
    )
    RE_HOTLINE = re.compile(r"(?is)\b(crisis\s+hotline|suicide\s+prevention\s+lifeline|samaritans|1-800-273|116\s+123|talk\s*\(8255\))\b")
    SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")

    @staticmethod
    def strip_markdown_emphasis(s: str) -> str:
        if not s: return s
        s = Stage4.RE_MD_BOLD.sub(r" \1 ", s)
        s = Stage4.RE_MD_BOLD_U.sub(r" \1 ", s)
        s = Stage4.RE_MD_ITALIC.sub(r" \1 ", s)
        s = Stage4.RE_MD_BARE_STARS.sub(" ", s)
        s = Stage4.RE_MD_BARE_UNDERS.sub(" ", s)
        s = re.sub(r"[ \t]{2,}", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    @staticmethod
    def strip_trailing_meta(text: str) -> str:
        if not text: return text
        t = text
        for _ in range(4):
            tail = t[-Stage4.TAIL_WINDOW:] if len(t) > Stage4.TAIL_WINDOW else t
            matches = list(Stage4.RE_TAIL_ANCHOR.finditer(tail))
            if not matches: break
            m = matches[-1]
            cut_pos = len(t) - len(tail) + m.start()
            post = t[cut_pos:]
            looks_meta = (
                Stage4.RE_ATTRIB.search(post) is not None
                or Stage4.RE_META_TAIL.search(post) is not None
                or (post.strip().lower().startswith("note:") and Stage4.RE_BULLETS.search(post) is not None)
            )
            if looks_meta: t = t[:cut_pos].rstrip()
            else: break
        return t

    @staticmethod
    def strip_leading_refusal(text: str) -> Tuple[Optional[str], str]:
        if not isinstance(text, str) or not text.strip(): return None, "drop"
        t = text.strip()
        head = t[:Stage4.LEAD_WINDOW]
        if not Stage4.RE_LEAD_APOLOGY.search(head): return t, "keep"
        if not (Stage4.RE_REFUSAL_CUE.search(head) or Stage4.RE_HOTLINE.search(head)): return t, "keep"
        if Stage4.RE_HOTLINE.search(head): return None, "drop"
        parts = [p.strip() for p in Stage4.SENT_SPLIT.split(t) if p.strip()]
        if not parts: return None, "drop"
        cut_idx = 0
        consumed = []
        for i, p in enumerate(parts[:3]):
            consumed.append(p)
            cut_idx = i + 1
            block = " ".join(consumed)
            if Stage4.RE_REFUSAL_CUE.search(block) or Stage4.RE_HOTLINE.search(block): continue
        remainder = " ".join(parts[cut_idx:]).strip()
        if not remainder: return None, "drop"
        if len(remainder) < 80: return None, "drop"
        rem_head = remainder[:Stage4.LEAD_WINDOW]
        if Stage4.RE_LEAD_APOLOGY.search(rem_head) and (Stage4.RE_REFUSAL_CUE.search(rem_head) or Stage4.RE_HOTLINE.search(rem_head)):
            return None, "drop"
        if Stage4.RE_REFUSAL_CUE.search(rem_head) and len(remainder) < 220:
            return None, "drop"
        return remainder, "strip"

    @staticmethod
    def clean(text: str) -> Optional[str]:
        t, _ = Stage4.strip_leading_refusal(text)
        if t is None: return None
        t = Stage4.strip_markdown_emphasis(t)
        t = Stage4.strip_trailing_meta(t)
        t = re.sub(r"[ \t]{2,}", " ", t)
        t = re.sub(r"\s*\n\s*", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if not t: return None
        return t

# ==============================================================================
# STAGE 5: clean_detectrl_dir5.py (Merged Patterns v2026-01-13)
# ==============================================================================
class Stage5:
    LEAD_WINDOW = 1200
    TAIL_WINDOW = 1800
    SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")
    CONFUSABLE_MAP = str.maketrans({
        "е": "e", "і": "i", "ѕ": "s", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
        "ԁ": "d", "һ": "h", "ӏ": "l", "ѵ": "v", "ɑ": "a", "ո": "n", "ɡ": "g", "ʋ": "v",
        "’": "'", "“": '"', "”": '"', "—": "-", "–": "-",
    })
    RE_MD_EMPH = re.compile(r"(\*\*|__)")
    RE_LEAD_STARSTAR = re.compile(r"(?s)^\s*(?:\*\s*){2,}")
    RE_POE_GUIDELINES = re.compile(r"(?is)\b(?:h?t{1,2}ps?)\s*:\s*/\s*/\s*poe\.com\s*/\s*usage_guidelines\b")
    RE_SEND_FEEDBACK = re.compile(r"(?is)\b(if\s+you\s+think\s+this\s+is\s+a\s+mistake|send\s+us\s+feedback|provide\s+us\s+with\s+your\s+feedback|contact\s+support)\b")
    RE_AI_IDENTITY = re.compile(r"(?is)\b(as\s+an?\s+ai(?:\s+language)?\s+model|i\s+am\s+(?:just|only|merely)\s+an?\s+ai(?:\s+language)?\s+model)\b")
    RE_REFUSAL_VERB = re.compile(r"(?is)\b(cannot|can'?t|won'?t|unable|not\s+able|must\s+decline|refuse)\b")
    RE_POLICY_CUE = re.compile(r"(?is)\b(policy|guidelines|ethical|not\s+appropriate|inappropriate|content\s+policy|safety)\b")
    RE_SELFHARM = re.compile(r"(?is)\b(self-?harm|suicid(?:e|al)|kill\s+yourself)\b")
    RE_IF_YOU_OR_SOMEONE = re.compile(r"(?is)\bif\s+you\s+or\s+someone\s+you\s+know\b")
    RE_REACH_OUT = re.compile(r"(?is)\b(reach\s+out|seek\s+help|contact|call)\b")
    RE_HOTLINE_KEY = re.compile(r"(?is)\b(hotline|lifeline|samaritans|crisis\s+line|emergency\s+services|988|116\s*123|1-800-273|call\s+911)\b")

    # Wrappers
    RE_ROLE_TAG_LINE = re.compile(r"(?im)^\s*(system|assistant|user)\s*[:：]\s*")
    RE_LEAD_BREAKDOWN = re.compile(r"(?is)^\s*(alright|okay|fine|sure)\b.{0,140}let'?s\s+break\s+this\s+down[.!]?\s*")
    RE_RULES_HEAD = re.compile(r"(?is)^\s*\[\s*(rules|auxiliary)\s*\]\s*:?\s*")
    RE_RULES_ANY = re.compile(r"(?is)\[\s*(rules|auxiliary)\s*\]\s*:?\s*")
    RE_LEAD_PROMPT_LABEL = re.compile(r"(?im)^\s*(prompt|writing\s*prompt|task|instruction|system\s*prompt|user\s*prompt)\s*:\s*")
    RE_LEAD_TASK_WRITE = re.compile(r"(?is)^\s*task\s*:\s*write\b.{0,360}?\bassistant\s*[:：]\s*")
    RE_INSTRUCTION_1 = re.compile(r"(?is)^\s*write\b.{0,280}\binstruction\s*1\s*:\s*")
    RE_INSTRUCTION_ANY_LINE = re.compile(r"(?im)^\s*[-*]?\s*instruction\s*\d+\s*:\s*")
    RE_LEAD_GIVEN_BASED = re.compile(r"(?is)^\s*(given\s+the|based\s+on\s+the)\b.{0,320}\b(abstract|story|essay|review|article|prompt)\b")
    RE_LEAD_WRITE_GEN = re.compile(r"(?is)^\s*(write|compose|generate|create|draft)\b.{0,180}\b(story|abstract|essay|review|article)\b")
    RE_LEAD_REVIEW_FIRST = re.compile(r"(?is)^\s*review[’']s\s+first\s+sentenc[e]\s*:\s*")
    RE_LEAD_GIVEN_REVIEW_FIRST = re.compile(r"(?is)^\s*given\s+the\s+review[’']s\s+first\s+sentenc[e]\s*,?\s*(?:continue|continu(?:e|ity))\b.{0,220}?:\s*")
    RE_LEAD_CONTINUED_REVIEW = re.compile(r"(?is)^\s*continued\s+review\s*:\s*")
    RE_LEAD_POSSIBLE_CONTINUATION = re.compile(r"(?is)^\s*(here'?s\s+a\s+possible|potential)\s+continuation\s+of\s+the\s+review\b.{0,220}:?\s*")
    RE_LEAD_FICTIONAL = re.compile(r"(?is)^\s*(here\s+is\s+a\s+fictional|this\s+is\s+a\s+fictional)\b.{0,140}:?\s*")
    RE_LEAD_NEWS_FIELDS = re.compile(r"(?im)^\s*(headline|body|date)\s*:\s*")
    RE_SENT_SCAFFOLD = re.compile(r"(?is)\b(generate an interesting title|write the (first|second|third|fourth|fifth|sixth|seventh) sentence)\b")
    RE_SCAFF_HERE_ARE = re.compile(r"(?is)\bhere\s+are\s+\d+\s+additional\s+sentences\b[^:]{0,160}:\s*")
    RE_NUM_BULLETS = re.compile(r"(?m)^\s*\d+\.\s+")
    RE_MEDIA_PLAYBACK = re.compile(r"(?is)\bmedia playback is not supported on this device\b")
    RE_LEAD_WARNING = re.compile(r"(?is)^\s*(trigger\s+warning|content\s+warning|tw\s*:)\b.{0,220}(\n\n|$)")

    # Tail
    RE_TAIL_ANCHOR = re.compile(
        r"(?is)("
        r"(?:^|[\n\r]|[.!?]\s+)\s*(?:---+\s*)?(?:editor'?s|author'?s|assistant'?s)?\s*note\s*:"
        r"|(?:^|[\n\r]|[.!?]\s+)\s*(?:---+\s*)?note\s*:"
        r"|(?:^|[\n\r]|[.!?]\s+)\s*(?:---+\s*)?(?:edit|update)\s*:"
        r"|(?:^|[\n\r]|[.!?]\s+)\s*(?:---+\s*)?tl\s*;\s*dr\s*[:\-]"
        r"|(?:^|[\n\r]|[.!?]\s+)\s*(?:---+\s*)?p\.?\s*s\.?\s*[:\-]"
        r"|(?:^|[\n\r]|[.!?]\s+)\s*(?:source|credit|attribution|link|original\s+prompt|original)\s*:"
        r"|(?:^|[\n\r]|[.!?]\s+)\s*(?:final\s+)?rating\s*:\s*\d"
        r"|(?:^|[\n\r]|[.!?]\s+)\s*year\s*:\s*\d{4}"
        r"|(?:^|[\n\r]|[.!?]\s+)\s*date\s*:\s*[A-Za-z]{3,}"
        r")"
    )
    RE_ATTRIB = re.compile(r"(?is)(https?://\S+|h?t{1,2}ps?://\S+|reddit\.com|r/writingprompts|(?:^|\s)/r/\w+|\b/u/\w+|poe\.com/usage_guidelines)")
    RE_BULLETS = re.compile(r"(?is)(?:^|\n)\s*(?:[*\-]\s+)")
    RE_META_TAIL = re.compile(
        r"(?is)\b("
        r"polished|rewritten|improved|grammar|spelling|clarity|concision|readability|"
        r"minor\s+changes?|made\s+some\s+changes?|rephrased|formatting|fixed\s+the\s+formatting|typo|"
        r"original\s+story|original\s+prompt|prompt\s+can\s+be\s+found|"
        r"can\s+be\s+found\s+on|the\s+link\s+to\s+the\s+original|"
        r"thanks\s+for\s+reading|feedback\s+welcome|send\s+us\s+feedback|if\s+you\s+think\s+this\s+is\s+a\s+mistake"
        r")\b"
    )
    RE_REFUSAL_START = re.compile(
        r"(?is)^\s*("
        r"i\s+apologize|i'?m\s+sorry|sorry|"
        r"i\s+cannot|i\s+can'?t|i\s+won'?t|"
        r"i\s+am\s+not\s+able|i\s+am\s+unable|"
        r"i\s+must\s+decline|i\s+have\s+to\s+decline|"
        r"regrettably|unfortunately"
        r")\b"
    )
    RE_REFUSAL_ACTION = re.compile(r"(?is)\b(help|assist|provide|generate|write|create|comply|fulfill|participate)\b")
    RE_NEGATIVE_NARRATIVE = re.compile(r"(?is)\b(can'?t\s+(?:believe|stand|wait|say|remember)|won'?t\s+(?:believe|say))\b")

    @staticmethod
    def normalize_text_base(t: str) -> str:
        t = unicodedata.normalize("NFKC", t)
        t = "".join(ch for ch in t if unicodedata.category(ch) != "Cf")
        t = t.translate(Stage5.CONFUSABLE_MAP)
        return t

    @staticmethod
    def normalize_ws(t: str) -> str:
        t = re.sub(r"[ \t\r\f\v]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()

    @staticmethod
    def should_drop_hard(head: str) -> bool:
        if Stage5.RE_POE_GUIDELINES.search(head): return True
        if Stage5.RE_SEND_FEEDBACK.search(head) and (Stage5.RE_AI_IDENTITY.search(head) or Stage5.RE_REFUSAL_VERB.search(head)): return True
        if Stage5.RE_IF_YOU_OR_SOMEONE.search(head) and Stage5.RE_SELFHARM.search(head) and (Stage5.RE_REACH_OUT.search(head) or Stage5.RE_HOTLINE_KEY.search(head)): return True
        if Stage5.RE_AI_IDENTITY.search(head) and Stage5.RE_REFUSAL_VERB.search(head) and Stage5.RE_POLICY_CUE.search(head): return True
        return False

    @staticmethod
    def strip_leading_prompt_task(t: str, min_keep_len: int) -> str:
        head = t[:900]
        m = Stage5.RE_LEAD_TASK_WRITE.match(t)
        if m:
            cand = t[m.end():].lstrip()
            if len(cand) >= min_keep_len: return cand
        if Stage5.RE_LEAD_PROMPT_LABEL.search(head):
            lines = t.splitlines()
            if len(lines) >= 2:
                cand = "\n".join(lines[1:]).lstrip()
                if len(cand) >= min_keep_len: return cand
        if Stage5.RE_LEAD_GIVEN_BASED.search(head):
            sep = t.find("\n\n")
            if 0 <= sep <= 2600:
                cand = t[sep + 2:].lstrip()
                if len(cand) >= min_keep_len: return cand
            parts = [p.strip() for p in Stage5.SENT_SPLIT.split(t) if p.strip()]
            if len(parts) >= 3:
                cand = " ".join(parts[2:]).strip()
                if len(cand) >= min_keep_len: return cand
        if Stage5.RE_LEAD_WRITE_GEN.search(head):
            lines = t.splitlines()
            if len(lines) >= 2 and len(lines[0].strip()) <= 280:
                cand = "\n".join(lines[1:]).lstrip()
                if len(cand) >= min_keep_len: return cand
        for rex in (Stage5.RE_LEAD_GIVEN_REVIEW_FIRST, Stage5.RE_LEAD_REVIEW_FIRST, Stage5.RE_LEAD_CONTINUED_REVIEW,
                    Stage5.RE_LEAD_POSSIBLE_CONTINUATION, Stage5.RE_LEAD_FICTIONAL):
            m2 = rex.match(t)
            if m2:
                cand = t[m2.end():].lstrip()
                if len(cand) >= min_keep_len: return cand
        if Stage5.RE_LEAD_NEWS_FIELDS.search(head):
            lines = t.splitlines()
            i = 0
            while i < min(len(lines), 10) and Stage5.RE_LEAD_NEWS_FIELDS.search(lines[i]): i += 1
            cand = "\n".join(lines[i:]).lstrip()
            if len(cand) >= min_keep_len: return cand
        # Instruction block logic
        if Stage5.RE_INSTRUCTION_1.search(head):
            lines = t.splitlines()
            last_idx = -1
            for i, line in enumerate(lines[:120]):
                if Stage5.RE_INSTRUCTION_ANY_LINE.search(line): last_idx = i
            if last_idx >= 0:
                cand = "\n".join(lines[last_idx + 1:]).lstrip()
                if len(cand) >= min_keep_len: return cand
        return t

    @staticmethod
    def strip_sentence_scaffold(t: str, min_keep_len: int) -> str:
        head = t[:900]
        t = Stage5.RE_LEAD_STARSTAR.sub("", t).lstrip()
        if Stage5.RE_SCAFF_HERE_ARE.search(head):
            t = Stage5.RE_SCAFF_HERE_ARE.sub("", t, count=1).lstrip()
            t = Stage5.RE_NUM_BULLETS.sub("", t)
        if Stage5.RE_SENT_SCAFFOLD.search(head):
            parts = [p.strip() for p in Stage5.SENT_SPLIT.split(t) if p.strip()]
            if len(parts) >= 2:
                cand = " ".join(parts[1:]).strip()
                if len(cand) >= min_keep_len: t = cand
        return t

    @staticmethod
    def strip_trailing_meta(t: str) -> str:
        if not t: return t
        s = t
        for _ in range(6):
            tail = s[-Stage5.TAIL_WINDOW:] if len(s) > Stage5.TAIL_WINDOW else s
            matches = list(Stage5.RE_TAIL_ANCHOR.finditer(tail))
            if not matches: break
            m = matches[-1]
            cut_pos = len(s) - len(tail) + m.start()
            post = s[cut_pos:]
            if Stage5.RE_ATTRIB.search(post) or Stage5.RE_META_TAIL.search(post) or Stage5.RE_BULLETS.search(post):
                s = s[:cut_pos].rstrip()
            else: break
        return s

    @staticmethod
    def handle_leading_refusal(t: str, min_keep_len: int) -> Optional[str]:
        if not isinstance(t, str) or not t.strip(): return None
        s = t.strip()
        head = s[:Stage5.LEAD_WINDOW]
        if Stage5.should_drop_hard(head): return None
        if Stage5.RE_LEAD_WARNING.search(head):
            idx = s.find("\n\n")
            if 0 <= idx <= 800:
                s2 = s[idx + 2:].lstrip()
                if len(s2) >= min_keep_len:
                    s = s2
                    head = s[:Stage5.LEAD_WINDOW]
        if not Stage5.RE_REFUSAL_START.search(head) and not Stage5.RE_AI_IDENTITY.search(head): return s
        if Stage5.RE_NEGATIVE_NARRATIVE.search(head) and not Stage5.RE_AI_IDENTITY.search(head): return s
        cue_ok = (Stage5.RE_AI_IDENTITY.search(head) or Stage5.RE_POLICY_CUE.search(head) or Stage5.RE_SEND_FEEDBACK.search(head))
        if not cue_ok and not (Stage5.RE_REFUSAL_ACTION.search(head) and Stage5.RE_REFUSAL_VERB.search(head)): return s
        parts = [p.strip() for p in Stage5.SENT_SPLIT.split(s) if p.strip()]
        if not parts: return None
        cut_idx = 0
        for i in range(min(6, len(parts))):
            cut_idx = i + 1
            block = " ".join(parts[:cut_idx])
            if (Stage5.RE_AI_IDENTITY.search(block) or Stage5.RE_POLICY_CUE.search(block) or Stage5.RE_SEND_FEEDBACK.search(block) or (Stage5.RE_REFUSAL_VERB.search(block) and Stage5.RE_REFUSAL_ACTION.search(block))): continue
            else: break
        remainder = " ".join(parts[cut_idx:]).strip()
        if not remainder or len(remainder) < min_keep_len: return None
        rem_head = remainder[:Stage5.LEAD_WINDOW]
        if (Stage5.RE_AI_IDENTITY.search(rem_head) and Stage5.RE_REFUSAL_VERB.search(rem_head)) or Stage5.should_drop_hard(rem_head): return None
        return remainder

    @staticmethod
    def is_mostly_nonword(s: str) -> bool:
        if not s: return True
        alnum = sum(ch.isalnum() for ch in s)
        return alnum < max(10, int(0.05 * len(s)))

    @staticmethod
    def clean_text_once(text: str, min_keep_len: int) -> Optional[str]:
        if not text.strip(): return None
        text = Stage5.normalize_text_base(text)
        text = Stage5.RE_MD_EMPH.sub(" ", text)
        text = Stage5.normalize_ws(text)
        if Stage5.RE_MEDIA_PLAYBACK.search(text):
            text = Stage5.RE_MEDIA_PLAYBACK.sub(" ", text)
            text = Stage5.normalize_ws(text)
        text = Stage5.handle_leading_refusal(text, min_keep_len)
        if text is None: return None

        # Rules Block
        s = text.lstrip()
        if Stage5.RE_RULES_HEAD.search(s[:220]):
            idx = s.find("\n\n")
            if 0 <= idx <= 2600: text = s[idx + 2:].lstrip()
            elif Stage5.RE_RULES_ANY.search(s[:1000]): text = s[1000:].lstrip()

        # Role tags
        if Stage5.RE_ROLE_TAG_LINE.search(text[:600]):
            text = Stage5.RE_ROLE_TAG_LINE.sub("", text).strip()
            text = Stage5.RE_LEAD_BREAKDOWN.sub("", text).strip()

        text = Stage5.strip_leading_prompt_task(text, min_keep_len)
        text = Stage5.strip_sentence_scaffold(text, min_keep_len)
        text = Stage5.strip_trailing_meta(text)
        text = Stage5.normalize_ws(text)
        if not text or len(text) < min_keep_len or Stage5.is_mostly_nonword(text): return None
        return text

    @staticmethod
    def clean(text: str, min_keep_len: int = 120, max_iter: int = 4) -> Optional[str]:
        cur = text
        for _ in range(max_iter):
            nxt = Stage5.clean_text_once(cur, min_keep_len)
            if nxt is None: return None
            if nxt == cur: return nxt
            cur = nxt
        return cur

# ==============================================================================
# STAGE 6: clean_detectrl_dir6.py (Pass-6 v2026-01-13)
# ==============================================================================
class Stage6:
    _CYR2LAT = str.maketrans({
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
        "і": "i", "ј": "j", "ѕ": "s", "ѵ": "v", "ӏ": "l",
        "ԁ": "d", "ԛ": "q", "ԝ": "w", "һ": "h", "к": "k", "м": "m", "н": "h", "т": "t",
        "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "Х": "X", "У": "Y",
        "І": "I", "Ј": "J", "Ѕ": "S", "Ѵ": "V", "Ӏ": "L",
        "Ԁ": "D", "Ԛ": "Q", "Ԝ": "W", "Н": "H", "К": "K", "М": "M", "Т": "T",
    })
    LEAD_WINDOW = 900
    TAIL_WINDOW = 1600
    MIN_KEEP_LEN = 100
    MIN_KEEP_AFTER_STRIP = 140
    SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")
    RE_MD_EMPH = re.compile(r"(\*\*|__)")
    RE_ROLE_LINE = re.compile(r"(?im)^\s*(?:#+\s*)?(system|assistant|user)\s*(?:[:\-]|：)\s*")
    RE_RULES_HEAD = re.compile(r"(?is)^\s*(\[\s*(rules|auxiliary)\s*\]|#{1,4}\s*(rules|auxiliary)\b|\*\*\s*(rules|auxiliary)\s*\*\*)")
    RE_LEAD_PROMPT_LABEL = re.compile(r"(?im)^\s*(prompt|writing\s*prompt|task|instruction|system\s*prompt)\s*[:：]\s*")
    RE_LEAD_FIELD_HEADER = re.compile(r"(?im)^\s*(title|abstract|keywords|subject|topic)\s*[:：]\s*")
    RE_LEAD_GIVEN_BASED = re.compile(r"(?is)^\s*(given\s+the|based\s+on\s+the)\b.{0,260}\b(story|abstract|essay|review|article|summary)\b")
    RE_LEAD_WRITE = re.compile(r"(?is)^\s*(write|compose|generate|create)\b.{0,160}\b(story|abstract|essay|review|article|summary|news)\b")
    RE_POLICY_CUE = re.compile(r"(?is)\b(policy|guidelines|content\s+policy|safety|as\s+an?\s+ai|language\s+model|artificial\s+intelligence|not\s+appropriate|inappropriate|ethical|morally|cannot\s+assist|can'?t\s+help|cannot\s+help|i\s+cannot\s+provide|i\s+can'?t\s+provide|request|prompt)\b")
    RE_REFUSAL_START = re.compile(r"(?is)^\s*(i\s+apologize|i'?m\s+sorry|sorry|i\s+cannot|i\s+can'?t|i\s+won'?t|i\s+am\s+not\s+able|i\s+am\s+unable|i\s+must\s+decline|i\s+have\s+to\s+decline|as\s+an?\s+ai|i\s+am\s+an?\s+artificial\s+intelligence)\b")
    RE_REFUSAL_LETS_FOCUS = re.compile(r"(?is)^\s*let'?s\s+focus\s+on\b")
    RE_REFUSAL_IF_STRUGGLING = re.compile(r"(?is)^\s*if\s+you\s+or\s+someone\s+you\s+know\s+is\s+(?:struggling|in\s+crisis)\b")
    RE_HOTLINE_STRICT = re.compile(r"(?is)\b(988\b|1[-\s]?800[-\s]?273\b|273[-\s]?talk\b|\b(?:8254|8255)\b|116\s*123\b|samaritans\b|crisis\s+hotline\b|suicide\s+(?:prevention|crisis)\b|(?:suicide|crisis|prevention)\s+\w{0,12}\s*lifeline\b)\b")
    RE_MEDIA_STRICT = re.compile(r"(?is)(\bwebvtt\b|\[music\]|\[applause\]|\u266a|like\s+and\s+subscribe|subscribe\s+to\s+our|closed\s+captions)")
    RE_TIME_STAMP = re.compile(r"(?m)\b\d{1,2}:\d{2}(?::\d{2})?\b")
    RE_TAIL_ANCHOR = re.compile(r"(?is)((?:editor'?s|author'?s|assistant'?s)\s*note\s*:|\"?\s*note\s*:|(?:edit|update)\s*:|tl\s*;\s*dr\s*[:\-] ง?|p\.?\s*s\.?\s*[:\-]|(?:source|credit|attribution|link|original\s+prompt)\s*:)")
    RE_ATTRIB = re.compile(r"(?is)(https?://\S+|reddit\.com|r/writingprompts|\b/u/\w+|(?:^|\s)/r/\w+)")
    RE_META_TAIL = re.compile(r"(?is)\b(polished|rewritten|improved|grammar|spelling|clarity|readability|minor\s+changes?|made\s+some\s+changes?|i\s+made\s+some\s+minor\s+changes?|rephrased|formatting|blank\s+lines|original\s+story|original\s+prompt|the\s+link\s+to\s+the\s+original|thanks\s+for\s+reading|feedback\s+welcome)\b")
    RE_CLOSING_HELP = re.compile(r"(?is)\b(i\s+hope\s+this\s+helps|let\s+me\s+know\s+if\s+you\s+have\s+any\s+other\s+questions|feel\s+free\s+to\s+ask)\b")

    @staticmethod
    def clean(text: str) -> Optional[str]:
        if not isinstance(text, str): return None
        # 0. Unicode/Confusable
        t = unicodedata.normalize("NFKC", text)
        t = t.translate(Stage6._CYR2LAT)
        # 1. Media Drop
        if Stage6.RE_MEDIA_STRICT.search(t[:2500]): return None
        if len(Stage6.RE_TIME_STAMP.findall(t[:2500])) >= 3 and re.search(r"(?is)\b(captions|subtitle|transcript)\b", t[:2500]): return None
        # 2. Markdown
        t = Stage6.RE_MD_EMPH.sub(" ", t)
        # 3. Rules
        l = t.lstrip()
        if Stage6.RE_RULES_HEAD.search(l[:250]):
            sep = l.find("\n\n")
            if 0 <= sep <= 2500: t = l[sep + 2:].lstrip()
            else: t = l[900:].lstrip() if len(l) > 900 else t
        # 4. Role Tags
        if Stage6.RE_ROLE_LINE.search(t[:500]): t = Stage6.RE_ROLE_LINE.sub("", t)
        # 5. Prompt Leakage
        l = t.lstrip()
        head = l[:800]
        if Stage6.RE_LEAD_PROMPT_LABEL.search(head):
            lines = l.splitlines()
            if len(lines) >= 2:
                cand = "\n".join(lines[1:]).lstrip()
                if len(cand) >= Stage6.MIN_KEEP_AFTER_STRIP: t = cand
        if Stage6.RE_LEAD_FIELD_HEADER.search(head):
            cand = Stage6.RE_LEAD_FIELD_HEADER.sub("", l, count=1).lstrip(" :-\t")
            if len(cand) >= Stage6.MIN_KEEP_AFTER_STRIP: t = cand
        if Stage6.RE_LEAD_GIVEN_BASED.search(head):
            sep = l.find("\n\n")
            if 0 <= sep <= 2500:
                cand = l[sep + 2:].lstrip()
                if len(cand) >= Stage6.MIN_KEEP_AFTER_STRIP: t = cand
            parts = [p.strip() for p in Stage6.SENT_SPLIT.split(l) if p.strip()]
            if len(parts) >= 3:
                cand = " ".join(parts[2:]).strip()
                if len(cand) >= Stage6.MIN_KEEP_AFTER_STRIP: t = cand
        if Stage6.RE_LEAD_WRITE.search(head):
            lines = l.splitlines()
            if len(lines) >= 2 and len(lines[0]) <= 260:
                cand = "\n".join(lines[1:]).lstrip()
                if len(cand) >= Stage6.MIN_KEEP_AFTER_STRIP: t = cand
        # 6. Refusal
        l = t.strip()
        head = l[:Stage6.LEAD_WINDOW]
        drop = False
        if Stage6.RE_HOTLINE_STRICT.search(head): drop = True
        elif Stage6.RE_REFUSAL_LETS_FOCUS.search(head) or Stage6.RE_REFUSAL_IF_STRUGGLING.search(head): drop = True
        else:
            if Stage6.RE_REFUSAL_START.search(head) and Stage6.RE_POLICY_CUE.search(head):
                parts = [p.strip() for p in Stage6.SENT_SPLIT.split(l) if p.strip()]
                if not parts: drop = True
                else:
                    cut_idx = 0
                    for i in range(min(7, len(parts))):
                        cut_idx = i + 1
                        block = " ".join(parts[:cut_idx])
                        if Stage6.RE_POLICY_CUE.search(block): continue
                        break
                    remainder = " ".join(parts[cut_idx:]).strip()
                    if not remainder or len(remainder) < Stage6.MIN_KEEP_AFTER_STRIP: drop = True
                    else:
                        if Stage6.RE_REFUSAL_START.search(remainder[:Stage6.LEAD_WINDOW]) and Stage6.RE_POLICY_CUE.search(remainder[:Stage6.LEAD_WINDOW]): drop = True
                        else: t = remainder
        if drop: return None

        # 7. Tail Meta
        for _ in range(4):
            tail = t[-Stage6.TAIL_WINDOW:] if len(t) > Stage6.TAIL_WINDOW else t
            m = None
            for mm in Stage6.RE_TAIL_ANCHOR.finditer(tail): m = mm
            if not m: break
            cut_pos = len(t) - len(tail) + m.start()
            post = t[cut_pos:]
            if Stage6.RE_ATTRIB.search(post) or Stage6.RE_META_TAIL.search(post): t = t[:cut_pos].rstrip()
            else: break

        # 8. Closing Help
        tail = t[-500:]
        if Stage6.RE_CLOSING_HELP.search(tail):
            parts = [p.strip() for p in Stage6.SENT_SPLIT.split(t) if p.strip()]
            if len(parts) > 2:
                if Stage6.RE_CLOSING_HELP.search(parts[-1]) and len(parts[-1]) <= 180:
                    cand = " ".join(parts[:-1]).strip()
                    if len(cand) >= Stage6.MIN_KEEP_LEN: t = cand

        # 9. WS
        t = re.sub(r"[ \t\r\f\v]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t).strip()

        if len(t) < Stage6.MIN_KEEP_LEN: return None
        return t

# ==============================================================================
# STAGE 7: clean_confusables_dir7.py (Confusables & Mixed Script)
# ==============================================================================
class Stage7:
    ASCII_LETTERS = re.compile(r"[A-Za-z]")
    TOKEN_RE = re.compile(r"\S+")

    ZERO_WIDTH = {0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF}
    BIDI = set(range(0x202A, 0x202F + 1)) | set(range(0x2066, 0x2069 + 1))

    CHAR_MAP = {
        "ⲅ": "r",   # COPTIC SMALL LETTER GAMMA often used as 'r'
        "հ": "h",   # ARMENIAN SMALL LETTER HO used as h
        "ς": "s",   # Greek final sigma -> s
        "Ь": "B",   # Cyrillic soft sign -> B
        "ь": "b",   # Cyrillic soft sign -> b
    }

    SUSPICIOUS_SCRIPTS = {"greek", "cyrillic", "armenian", "coptic"}

    @staticmethod
    def is_invisible_control(ch: str) -> bool:
        cp = ord(ch)
        return (cp in Stage7.ZERO_WIDTH) or (cp in Stage7.BIDI)

    @staticmethod
    def is_letter(ch: str) -> bool:
        return unicodedata.category(ch).startswith("L")

    @staticmethod
    def script_bucket(ch: str) -> str:
        if ord(ch) < 128: return "ascii"
        name = unicodedata.name(ch, "")
        if "LATIN" in name and "LETTER" in name: return "latin"
        if name.startswith("GREEK"): return "greek"
        if name.startswith("CYRILLIC"): return "cyrillic"
        if name.startswith("ARMENIAN"): return "armenian"
        if name.startswith("COPTIC"): return "coptic"
        return "other"

    @staticmethod
    def token_has_suspicious_letter(tok: str) -> bool:
        for ch in tok:
            if ord(ch) < 128: continue
            if not Stage7.is_letter(ch): continue
            if Stage7.script_bucket(ch) in Stage7.SUSPICIOUS_SCRIPTS: return True
        return False

    @staticmethod
    def is_mixed_script_token(tok: str) -> bool:
        return bool(Stage7.ASCII_LETTERS.search(tok)) and Stage7.token_has_suspicious_letter(tok)

    @staticmethod
    def fix_token(tok: str) -> str:
        # 1. Remove invisibles
        if any(Stage7.is_invisible_control(c) for c in tok):
            tok = "".join(c for c in tok if not Stage7.is_invisible_control(c))

        # 2. Normalize and replace chars
        tok_nfkc = unicodedata.normalize("NFKC", tok)
        out = []
        for c in tok_nfkc:
            if c in Stage7.CHAR_MAP: out.append(Stage7.CHAR_MAP[c])
            else: out.append(c)
        return "".join(out)

    @staticmethod
    def clean(text: str) -> Optional[str]:
        if not isinstance(text, str) or not text: return text

        # 1. Global invisible removal
        if any(Stage7.is_invisible_control(c) for c in text):
            text = "".join(c for c in text if not Stage7.is_invisible_control(c))

        # 2. Fast exit if no non-ascii
        if not any(ord(c) >= 128 for c in text): return text

        # 3. Token-wise replacement
        def repl(m: re.Match) -> str:
            tok = m.group(0)
            if Stage7.is_mixed_script_token(tok):
                return Stage7.fix_token(tok)
            return tok

        new_text = Stage7.TOKEN_RE.sub(repl, text)
        return new_text

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description="DetectRL Merged Cleaning Pipeline (Dir1-Dir7)")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--fields", nargs="+", default=["text"])
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    files = [p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in [".json", ".jsonl"]]

    print(f"Processing {len(files)} files via merged pipeline (Stages 1-7)...")

    total_kept = 0
    total_dropped = 0

    for src in files:
        rel = src.relative_to(in_dir)
        dst = out_dir / rel

        raw_obj, recs = read_records_raw(src)
        out_recs = []
        file_drop = 0
        file_in = len(recs)

        for r in recs:
            if not isinstance(r, dict):
                out_recs.append(r)
                continue

            ok = True
            for f in args.fields:
                if f in r and isinstance(r[f], str):
                    t = r[f]

                    # --- PIPELINE EXECUTION ---
                    # 1. Gold (Wrappers/Tags)
                    t = Stage1.clean(t)
                    if t is None: ok = False; break

                    # 2. Black Hole (Recursive/Ordinals)
                    t = Stage2.clean(t)
                    if t is None: ok = False; break

                    # 3. Tail Cutter (Simple)
                    t = Stage3.clean(t)

                    # 4. Leading Refusal / Markdown / Tail V13
                    t = Stage4.clean(t)
                    if t is None: ok = False; break

                    # 5. Merged Patterns / Strong Drops / Aggressive Tail
                    t = Stage5.clean(t, min_keep_len=120, max_iter=4)
                    if t is None: ok = False; break

                    # 6. Final Polish / Strict Hotline / Confusables
                    t = Stage6.clean(t)
                    if t is None: ok = False; break

                    # 7. Confusables / Homoglyphs / Invisibles
                    t = Stage7.clean(t)
                    # Stage 7 does not return None (drop), so no check needed.

                    r[f] = t

            if ok:
                out_recs.append(r)
                total_kept += 1
            else:
                file_drop += 1
                total_dropped += 1

        write_records(dst, raw_obj, out_recs)
        print(f"[OK] {rel} | In: {file_in} | Out: {len(out_recs)} | Drop: {file_drop}")

    print("="*60)
    print(f"PIPELINE COMPLETE.")
    print(f"Total Kept: {total_kept}")
    print(f"Total Dropped: {total_dropped}")
    print(f"Output: {out_dir}")

if __name__ == "__main__":
    main()