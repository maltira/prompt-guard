"""разбор письма: видимый и скрытый текст

детекторы получают тему и тело в том виде, в каком их обучали и оценивали (сырой текст с разметкой),
а разбор нужен, чтобы показать, какая часть письма скрыта от человека, но видна ассистенту
"""
import re
from dataclasses import dataclass
from html.parser import HTMLParser

HTML_RE = re.compile(r"<\s*(!--|/?[a-zA-Z][a-zA-Z0-9]*)")
VOID_TAGS = {"br", "hr", "img", "input", "meta", "link", "area", "base", "col", "embed", "source", "track", "wbr"}
SKIP_TAGS = {"script", "style"}
HIDDEN_COLORS = {"white", "#fff", "#ffffff", "transparent", "rgb(255,255,255)", "rgba(255,255,255,1)"}
NUMBER_RE = re.compile(r"^(-?\d*\.?\d+)([a-z%]*)$")


def _number(value: str) -> tuple[float, str] | None:
    m = NUMBER_RE.match(value)
    return (float(m.group(1)), m.group(2)) if m else None


def style_hides(style: str) -> bool:
    """признаки того, что текст с таким стилем человек не увидит"""
    decl = {}
    for part in re.sub(r"\s+", "", style.lower()).split(";"):
        if ":" in part:
            key, value = part.split(":", 1)
            decl[key] = value.replace("!important", "")
    if decl.get("display") == "none" or decl.get("visibility") in ("hidden", "collapse"):
        return True
    opacity = _number(decl.get("opacity", ""))
    if opacity and opacity[0] <= 0.01:
        return True
    size = _number(decl.get("font-size", ""))
    if size and (size[0] == 0 or (size[1] in ("px", "pt") and size[0] <= 2)):
        return True
    color = decl.get("color", "")
    if color in HIDDEN_COLORS or re.fullmatch(r"rgba\(\d+,\d+,\d+,0(\.0+)?\)", color):
        return True  # белый на светлом фоне: типичный приём, фон не проверяется
    for key in ("text-indent", "left", "top", "margin-left"):
        offset = _number(decl.get(key, ""))
        if offset and offset[0] <= -999:
            return True
    collapsed = any((_number(decl.get(k, "")) or (1, ""))[0] == 0 for k in ("height", "max-height"))
    return collapsed and decl.get("overflow") == "hidden"


class _Collector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool]] = []  # (тег, скрыт ли он вместе с потомками)
        self.segments: list[tuple[str, bool]] = []

    def handle_starttag(self, tag, attrs):
        if tag in VOID_TAGS:
            return
        attr = dict(attrs)
        own = "hidden" in attr or style_hides(attr.get("style") or "")
        self.stack.append((tag, own or (self.stack[-1][1] if self.stack else False)))

    def handle_startendtag(self, tag, attrs):
        pass  # пустой тег без содержимого

    def handle_endtag(self, tag):
        names = [t for t, _ in self.stack]
        if tag in names:
            del self.stack[len(names) - names[::-1].index(tag) - 1:]

    def handle_data(self, data):
        if any(t in SKIP_TAGS for t, _ in self.stack):
            return
        text = " ".join(data.split())
        if text:
            self.segments.append((text, self.stack[-1][1] if self.stack else False))

    def handle_comment(self, data):
        text = " ".join(data.split())
        if text:
            self.segments.append((text, True))  # комментарии в письме не видны, но остаются в исходнике


@dataclass
class ParsedMail:
    subject: str
    visible_text: str
    hidden_text: str
    is_html: bool

    @property
    def hidden_found(self) -> bool:
        return bool(self.hidden_text)


def parse_mail(subject: str, body: str) -> ParsedMail:
    if not HTML_RE.search(body):
        return ParsedMail(subject.strip(), " ".join(body.split()), "", False)
    collector = _Collector()
    collector.feed(body)
    collector.close()
    visible = " ".join(t for t, hidden in collector.segments if not hidden)
    hidden = " ".join(t for t, hidden in collector.segments if hidden)
    return ParsedMail(subject.strip(), visible, hidden, True)


def compose_text(subject: str, body: str) -> str:
    """текст для детектора: тема и тело в формате обучающих писем"""
    subject = subject.strip()
    if not subject:
        return body
    label = "Тема" if re.search("[а-яё]", subject + body, re.IGNORECASE) else "Subject"
    return f"{label}: {subject}\n\n{body}"
