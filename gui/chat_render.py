"""Pure helpers for the chat page (no NiceGUI): source records, clickable [n] citations, and turning the
captured terminal output of /compare and /reading into Markdown. Kept separate so they are easy to test."""
import html
import re

LABEL_RE = re.compile(r"^(?P<title>.*) \((?P<who>[^()]*), (?P<year>\d{4})\), p\.(?P<page>\d+)"
                      r"(?:, (?P<section>.*?))?\s+\[(?P<key>[^\]\s]+)\]\s*$")


def first_author(authors):
    authors = authors or ""
    return authors.split(",")[0].strip() + (" et al." if "," in authors else "")


def source_from_hit(papers, h):
    """A retrieved passage -> the record the page shows (title, author, year, page, section, key)."""
    p = papers.get(h["paper_key"], {})
    return dict(key=h["paper_key"], title=p.get("title") or h["paper_key"], who=first_author(p.get("authors")),
                year=p.get("year") or "", page=h.get("page_start"), page_end=h.get("page_end"),
                section=h.get("section") or "")


def source_from_label(label):
    """A saved source label (ask.source_label) -> the same record; used for resumed conversations."""
    m = LABEL_RE.match(label.strip())
    if not m:
        key = label.rsplit("[", 1)[-1].rstrip("] ") if label.rstrip().endswith("]") else ""
        return dict(key=key, title=label, who="", year="", page=None, page_end=None, section="")
    d = m.groupdict()
    return dict(key=d["key"], title=d["title"], who=d["who"], year=d["year"], page=int(d["page"]),
                page_end=None, section=d["section"] or "")


def source_tooltip(s):
    bits = [s["title"]]
    if s.get("year"):
        bits[0] += f" ({s['year']})"
    if s.get("page"):
        bits.append(f"p.{s['page']}")
    if s.get("section"):
        bits.append(s["section"])
    return " - ".join(bits)


_FENCE = re.compile(r"(```.*?(?:```|$))", re.S)
_INLINE_CODE = re.compile(r"(`[^`\n]*`)")
_CITE_GROUP = re.compile(r"\[(\d{1,3}(?:\s*(?:,|;|-|–)\s*\d{1,3})*)\](?!\()")
_NUM = re.compile(r"\d{1,3}")


_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


def fix_lists(md):
    """markdown2 only starts a list after a blank line, but Claude often writes '**Heading**\\n- item'.
    Insert the blank line (outside code fences) so such lists render as lists."""
    out, prev, fence = [], "", False
    for line in md.split("\n"):
        if line.lstrip().startswith("```"):
            fence = not fence
        elif (not fence and _LIST_ITEM.match(line) and prev.strip() and not _LIST_ITEM.match(prev)
              and not prev.startswith((" ", "\t")) and not prev.lstrip().startswith("|")):
            out.append("")
        out.append(line)
        prev = line
    return "\n".join(out)


def linkify_citations(md, answer_id, sources):
    """[n], [1, 3], [2-4] -> clickable links (class "cite") that the page turns into 'open the PDF at this
    source's page'. Numbers outside the source list and anything inside code are left alone."""
    if not sources:
        return md

    def group(m):
        def num(nm):
            n = int(nm.group(0))
            if not 1 <= n <= len(sources):
                return nm.group(0)
            tip = html.escape(source_tooltip(sources[n - 1]), quote=True)
            return f'<a href="#" class="cite" data-a="{answer_id}" data-n="{n}" title="{tip}">{n}</a>'
        if not any(1 <= int(x) <= len(sources) for x in _NUM.findall(m.group(1))):
            return m.group(0)
        compact = re.sub(r"\s*[,;]\s*", " ", m.group(1))          # [1, 3] -> two pills side by side
        compact = re.sub(r"\s*[-–]\s*", "–", compact)             # ranges keep their dash
        return f'<span class="cite-group">{_NUM.sub(num, compact)}</span>'

    out = []
    for i, part in enumerate(_FENCE.split(md)):
        if i % 2:                                     # fenced code block
            out.append(part)
            continue
        out.append("".join(p if j % 2 else _CITE_GROUP.sub(group, p)
                           for j, p in enumerate(_INLINE_CODE.split(part))))
    return "".join(out)


# ------------------------------------------------------------ /compare and /reading output

_TRAILER_START = re.compile(r"^(Sources:|Candidates:|Comparing \d+ papers:)\s*$")
_TRAILER_ITEM = re.compile(r"^(?P<indent>\s*)(?P<tag>\[\d+\]|P\d+(?:\.\d+)?)\s+(?P<rest>.*)$")
_KEY = re.compile(r"\[(v\d+/[^\]\s]+)\]")
_PROGRESS = re.compile(r"^(Stage \d+:|Found \d+ candidate|Retrieved \d+ passages)")
_COST = re.compile(r"^\(.*(tokens|~\$).*\)$")


def oneshot_markdown(text):
    """Captured stdout of ask.cmd_compare / ask.cmd_reading -> Markdown: progress and cost lines in italics,
    the candidate/source lists as Markdown lists with links to the paper pages, the answer itself unchanged."""
    out, trailer = [], False
    for line in text.replace("\r", "").split("\n"):
        s = line.strip()
        if s.startswith("Open a paper with:"):
            continue
        if _TRAILER_START.match(s):
            trailer = True
            out += ["", f"**{s.rstrip(':')}**" if not s.startswith("Comparing") else f"*{s}*", ""]
            continue
        if trailer:
            m = _TRAILER_ITEM.match(line)
            if m:
                rest = _KEY.sub(lambda k: f"[[{k.group(1)}]](/paper/{k.group(1)})", m.group("rest"))
                depth = "  " if len(m.group("indent")) > 3 else ""
                out.append(f"{depth}- **{m.group('tag')}** {rest}")
                continue
            if not s:
                trailer = False
                out.append("")
                continue
            trailer = False
            out.append("")
        if _PROGRESS.match(s) or _COST.match(s):
            out += ["", f"*{s}*", ""]
            continue
        out.append(line)
    md = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", md).strip()
