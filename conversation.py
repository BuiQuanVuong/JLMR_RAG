"""Conversational memory for `ask.py chat`.

A Conversation keeps the finished turns (question, search query, answer text, source labels). Two things
use it:

1. condense(): a follow-up like "how does the second one handle noise?" is rewritten by Claude Haiku 5.5
   into a standalone search query, using the last few turns (answers and numbered source titles). If the
   call fails (no key, network, refusal...) the raw question is used.
2. history_messages(): the earlier turns as real user/assistant messages for the answer call. Old turns
   carry only their question and a compact list of source labels (not the passages), and every turn is
   tagged "Turn N", so the model can tell that an earlier [n] meant that turn's source list.

Retrieval itself (Index.passages) is untouched: it simply receives the condensed query.
"""
import json
from dataclasses import dataclass, field

CONDENSE_MODEL = "claude-haiku-5-5"
HISTORY_TURNS = 4          # at most this many earlier turns go into the answer call ...
HISTORY_CHARS = 24000      # ... and at most ~this many characters of them (~6k tokens)
ANSWER_CHARS = 6000        # a single earlier answer is cut to this length in the history
CONDENSE_TURNS = 3         # turns shown to the condenser
CONDENSE_ANSWER_CHARS = 1500

CHAT_SYSTEM_ADDENDUM = """

This is a multi-turn conversation. Each turn ("Turn N") has its own numbered source list, so [n] in an \
earlier turn refers to that turn's list, not to the current one. Earlier turns show only the source \
titles; the current turn includes the full passages.
- Cite only the sources in the current <sources> block with [n]. Never reuse a citation number from an \
earlier turn.
- To refer to a paper discussed earlier, name it (short title or first author + year). If it also appears \
in the current sources, cite its current number; if not, say that its passages are not in the current \
sources.
- Resolve references like "the second paper" or "that method" using the earlier turns."""

CONDENSE_PROMPT = """You rewrite follow-up questions in a conversation about Journal of Machine Learning \
Research (JMLR) papers into standalone search queries for a retrieval system (keyword + semantic search \
over paper passages).

Conversation so far (each turn lists the sources it used; "[n]" in an answer refers to that turn's list):

{history}

Follow-up question: {question}

Write one standalone search query that finds passages answering the follow-up:
- Resolve pronouns and references ("it", "that method", "the second paper", "source 3", "those papers") \
using the conversation; include the specific paper title(s), method name(s) or topic they refer to.
- Keep the technical terms from the follow-up and add the key topic terms from the conversation that are \
needed to understand it.
- If the follow-up is already standalone or starts a new topic, return it essentially unchanged.
- At most about 30 words, no explanations."""

CONDENSE_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}


@dataclass
class Turn:
    question: str
    query: str                      # what was actually searched
    answer: str
    sources: list = field(default_factory=list)   # labels, e.g. "Title (2012), p.4, Section [v13/x]"


@dataclass
class Conversation:
    turns: list = field(default_factory=list)

    def reset(self):
        self.turns.clear()

    def add(self, question, query, answer, sources):
        self.turns.append(Turn(question, query, answer, list(sources)))

    def __len__(self):
        return len(self.turns)

    # ------------------------------------------------------------ answer-call history

    def history_messages(self, max_turns=HISTORY_TURNS, max_chars=HISTORY_CHARS):
        """Earlier turns as alternating user/assistant messages (oldest first), newest turns kept first
        when the budget runs out."""
        picked, used = [], 0
        start = len(self.turns)
        for t in reversed(self.turns[-max_turns:]):
            n = start - len(picked)            # 1-based turn number
            user, assistant = self._user_text(n, t), clip(t.answer, ANSWER_CHARS) or "(no answer)"
            size = len(user) + len(assistant)
            if picked and used + size > max_chars:
                break
            picked.append((user, assistant))
            used += size
        msgs = []
        for user, assistant in reversed(picked):
            msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        return msgs

    @staticmethod
    def _user_text(n, t):
        src = "\n".join(f"[{i}] {s}" for i, s in enumerate(t.sources, 1))
        return (f"Turn {n}. Sources for this turn (passages omitted from the history):\n{src}\n\n"
                f"Question: {t.question}")

    def current_label(self):
        return f"Turn {len(self.turns) + 1}"

    # ------------------------------------------------------------ query condensation

    def condense(self, question, client=None):
        """Standalone search query for a follow-up. Returns (query, cost_usd, error_or_None)."""
        if not self.turns:
            return question, 0.0, None
        lines = []
        recent = self.turns[-CONDENSE_TURNS:]
        for i, t in enumerate(recent, len(self.turns) - len(recent) + 1):
            src = "\n".join(f"  [{j}] {s}" for j, s in enumerate(t.sources, 1))
            lines.append(f"Turn {i} question: {t.question}\nTurn {i} sources:\n{src}\n"
                         f"Turn {i} answer: {clip(t.answer, CONDENSE_ANSWER_CHARS)}")
        prompt = CONDENSE_PROMPT.format(history="\n\n".join(lines), question=question)
        try:
            if client is None:
                import anthropic
                client = anthropic.Anthropic()
            msg = client.messages.create(
                model=CONDENSE_MODEL, max_tokens=1000,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": CONDENSE_SCHEMA}},
                messages=[{"role": "user", "content": prompt}])
            u = msg.usage
            cost = (u.input_tokens * 0.10 + u.output_tokens * 0.50) / 1e6
            if msg.stop_reason != "end_turn":
                return question, cost, f"stop_reason={msg.stop_reason}"
            text = next(b.text for b in msg.content if b.type == "text")
            query = " ".join(json.loads(text)["query"].split())
            return (query or question), cost, None
        except Exception as e:        # no key, network, bad JSON...: search with the raw question
            return question, 0.0, f"{type(e).__name__}: {str(e)[:120]}"


def clip(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " [...]"
