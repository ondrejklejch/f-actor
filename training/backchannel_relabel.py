"""Re-derives an example's backchannel labels from word-level alignments, instead of trusting
whatever `utterances[*].backchannels` the dataset ships.

Standalone: only touches the `utterances` list/dict shape shared by every dataset_format in this
repo (a list of {speaker_idx, start_time, end_time, tts_text, words, backchannels}), and returns a
new list in the exact same shape, so callers don't need to know relabeling happened.

Usage:
    from backchannel_relabel import relabel_backchannels
    example["utterances"] = relabel_backchannels(example["utterances"])

Wired into training via DataArgs.relabel_backchannels (training/arguments/arguments.py) - see
load_speech_data's tokenize_speech in training/data.py, which applies this before prepare_dsu/
adapt_to_text_stream ever see the example, when that flag is set.
"""

import bisect
import re
from typing import NamedTuple

BACKCHANNEL_REGEXES = {
    # The "Yeah" family (yeah, yeeeah, yah, yea)
    "yeah": r"\by+e*a+h*\b",

    # The "Yes" family (yes, yesss)
    "yes": r"\by+e+s+\b",

    # The "Yep/Yup" family (yep, yeeep, yup, yuuup)
    "yep_yup": r"\by+[eu]+p+\b",

    # Nasal agreements (mhm, mmhmm, mm-hmm)
    "mhm": r"\bm+[\s-]*h+m+\b",

    # The "Huh" family (huh, huuuuh)
    "huh": r"\bh+u+h+\b",

    # Standard words with drawn-out endings (riiight, righttt)
    "right": r"\br+i+g+h+t+\b",

    # The "Okay" family (ok, okay, okkk, okaaay)
    "okay": r"\bo+k+(?:a+y+)?\b",

    # Exclamations and fillers (ohhh, ooooh)
    "oh": r"\bo+h+\b",

    # Thinking noises (hm, hmm, hmmm, hum, hummm)
    "hm_hum": r"\bh+u*m+\b",

    # Standard fillers (uhh, uuuuh, umm, uuum)
    "uh": r"\bu+h+\b",
    "um": r"\bu+m+\b",
    "ah": r"\ba+h+\b",

    # Two-syllable grunts (uhhuh, uh-huh, umhum, um-hum)
    "uhhuh_umhum": r"\bu+[hm]+[\s-]*h+u+[hm]+\b",

    # Reactions & Agreements (wooow, suuure, exactlly, agreed, indeed)
    "wow": r"\bw+o+w+\b",
    "sure": r"\bs+u+r+e+\b",
    "exactly": r"\be+x+a+c+t+l+y+\b",
    "agreed": r"\ba+g+r+e+e+d*\b",
    "indeed": r"\bi+n+d+e+e+d+\b",
    "absolutely": r"\ba+b+s+o+l+u+t+e+l+y+\b",
    "definitely": r"\bd+e+f+i+n+i+t+e+l+y+\b",
    "totally": r"\bt+o+t+a+l+l+y+\b",
}

# Additions on top of the base lexicon, drawn from the most frequent short units that overlap
# the partner yet failed the patterns above.
BACKCHANNEL_REGEXES_EXTRA = {
    # Bare nasal. The base lexicon's two nasal patterns both require an 'h' ("mhm", "hum"), so
    # plain "mm" matched neither.
    "mm": r"\bm+\b",

    # "really" / "oh really" is a common backchannel and was absent.
    "really": r"\br+e+a+l+l*y+\b",

    # "aha" (`a+h+` alone stops at "ah"), plus assorted assessments.
    "aha": r"\ba+h+a+\b",
    "cool": r"\bc+o+o+l+\b",
    "nice": r"\bn+i+c+e+\b",
    "gotcha": r"\bg+o+t+c+h+a+\b",
    "whoa": r"\bw+h*o+a+h*\b",

    # Filler variants alongside uh/um/ah.
    "er_em": r"\b[eE]+[rm]+\b",
}

SINGLE_WORD_BC_REGEX = re.compile(
    r"^(?:" + "|".join(list(BACKCHANNEL_REGEXES.values()) + list(BACKCHANNEL_REGEXES_EXTRA.values())) + r")$",
    re.IGNORECASE,
)
# lexicon="base" selects the base patterns alone, with no extra words and no phrases.
SINGLE_WORD_BC_REGEX_BASE = re.compile(
    r"^(?:" + "|".join(BACKCHANNEL_REGEXES.values()) + r")$",
    re.IGNORECASE,
)
LEXICONS = ("base", "extended")

# Multi-word backchannels. These have to be matched as phrases rather than by adding their parts
# to the single-word list: "i see" is a backchannel but a bare "i" is an aborted turn start, and
# "no kidding" is one while a bare "no" is usually an answer.
BACKCHANNEL_PHRASES = (
    "i see",
    "i know",
    "i bet",
    "that's right",
    "that's true",
    "you bet",
    "no kidding",
    "no way",
    "right on",
    "oh my god",
    "oh my gosh",
    "for sure",
    "makes sense",
    "good point",
    "fair enough",
)


def _clean_word(word: str) -> str:
    return "".join(c for c in word.lower() if c.isalnum())


# Phrases stored in the same alnum-only form is_backchannel_word compares in, so the literals above
# stay readable ("that's right") while matching sees what the cleaner produces ("thats right").
_CLEAN_PHRASES = frozenset(" ".join(_clean_word(w) for w in p.split()) for p in BACKCHANNEL_PHRASES)
_MAX_PHRASE_WORDS = max(len(p.split()) for p in BACKCHANNEL_PHRASES)


def is_backchannel_word(word: str, lexicon: str = "extended") -> bool:
    """Returns True if a single word (stripped of punctuation) matches any backchannel pattern."""
    cleaned = _clean_word(word)
    if not cleaned:
        return True  # ignore purely punctuation tokens
    pattern = SINGLE_WORD_BC_REGEX if lexicon == "extended" else SINGLE_WORD_BC_REGEX_BASE
    return bool(pattern.match(cleaned))


def is_backchannel_text(text: str, lexicon: str = "extended") -> bool:
    """True if `text` consists entirely of backchannel words and phrases.

    Consumes the words left to right, taking the longest phrase match available at each position
    and otherwise requiring a single-word match. That composes: "oh i see" is oh + [i see], while
    "i think so" fails on the bare "i". lexicon="base" disables both the extra words and the
    phrases, reducing this to an all-words-match test.
    """
    text_no_punct = re.sub(r'[\,\.\!\?\:\;\)\(\[\]"\-\~]', ' ', text)
    words = text_no_punct.strip().split()
    if not words:
        return False

    i = 0
    while i < len(words):
        if lexicon == "extended":
            for span in range(min(_MAX_PHRASE_WORDS, len(words) - i), 1, -1):
                if " ".join(_clean_word(w) for w in words[i : i + span]) in _CLEAN_PHRASES:
                    i += span
                    break
            else:
                if not is_backchannel_word(words[i], lexicon):
                    return False
                i += 1
        else:
            if not is_backchannel_word(words[i], lexicon):
                return False
            i += 1
    return True


# Some transcripts carry non-speech event tokens ([NOISE], [LAUGHTER], [SIGH], ...). These have
# to be dropped *before* the lexical test rather than left to is_backchannel_text, whose
# punctuation stripping turns "[LAUGHTER]" into the plain word "LAUGHTER" and thus fails the
# all-words-are-backchannels check. Dropping them means "[LAUGHTER] YEAH" still counts.
_NON_SPEECH_TOKEN = re.compile(r"^\[[^\]]*\]$")

# Laughter is the one non-speech event that functions as a backchannel.
_LAUGHTER_TOKEN = "[laughter]"


def is_non_speech(word: str) -> bool:
    """True for a bracketed non-speech event token ("[noise]", "[laughter]", ...)."""
    return bool(_NON_SPEECH_TOKEN.match(word.strip()))


class Ipu(NamedTuple):
    """One inter-pausal unit: a run of one speaker's words with no internal pause >= ipu_pause."""

    speaker_idx: int
    start: float
    end: float
    text: str
    is_backchannel: bool
    # Provenance of the IPU's first (lexical) word within the `utterances` list passed in.
    utt_idx: int
    word_idx: int
    # "lexical" | "laughter" | None -- which gate accepted it, for reporting.
    kind: "str | None" = None
    # True when this unit is a backchannel that its own speaker follows with a real turn inside
    # turn_clearance seconds - still a backchannel acoustically, but not a clean <bc> training
    # example (it's "acknowledge, then take the floor").
    precedes_own_turn: bool = False


def speaker_words(utterances, speaker_idx):
    """This speaker's words, chronologically, as (start, end, text, utt_idx, word_idx).

    Sorted explicitly: `utterances` may be grouped by channel rather than time-ordered.
    """
    words = []
    for utt_idx, utt in enumerate(utterances):
        if utt["speaker_idx"] != speaker_idx:
            continue
        for word_idx, word in enumerate(utt["words"]):
            start = word["start"]
            words.append((start, start + word["dur"], word["word"], utt_idx, word_idx))
    words.sort(key=lambda w: w[0])
    return words


def _build_ipus(utterances, speaker_idx, ipu_pause):
    """Split one speaker's word stream at every pause >= ipu_pause."""
    ipus = []
    current = []
    for word in speaker_words(utterances, speaker_idx):
        if current and word[0] - current[-1][1] >= ipu_pause:
            ipus.append(current)
            current = []
        current.append(word)
    if current:
        ipus.append(current)
    return ipus


class _IntervalSet:
    """Sorted intervals with a prefix maximum of end times, so `overlaps` is O(log n).

    An interval j overlaps [start, end) iff j.start < end and j.end > start (strictly positive
    intersection: merely touching endpoints does not count). Bisect bounds the first condition;
    the prefix max of end times answers the second over that whole range at once.
    """

    def __init__(self, intervals):
        intervals = sorted(intervals)
        self._starts = [s for s, _ in intervals]
        self._prefix_max_end = []
        running = float("-inf")
        for _, end in intervals:
            running = max(running, end)
            self._prefix_max_end.append(running)

    def overlaps(self, start, end):
        cutoff = bisect.bisect_left(self._starts, end)
        return cutoff > 0 and self._prefix_max_end[cutoff - 1] > start


def _held_floor(ordered, index, floor_gap):
    """True if the opposing speaker was talking just before this IPU and resumes just after.

    The complement of the overlap test rather than a replacement for it: overlap catches a
    backchannel uttered *inside* the partner's ongoing speech; this catches one dropped into a
    short gap, which is just as much a backchannel - what defines one is that it doesn't take
    the floor.
    """
    speaker = ordered[index].speaker_idx
    before = next((o for o in reversed(ordered[:index]) if o.speaker_idx != speaker), None)
    after = next((o for o in ordered[index + 1 :] if o.speaker_idx != speaker), None)
    if before is None or after is None:
        return False
    return (
        ordered[index].start - before.end < floor_gap
        and after.start - ordered[index].end < floor_gap
    )


def _precedes_own_turn(ordered, index, clearance):
    """True if this speaker's next unit starts within `clearance` seconds and is not itself a
    backchannel.

    Measured to the next unit's *start*, i.e. the silence the model would have to produce before
    speaking again. A following backchannel does not count: "yeah. yeah." is one acknowledgement
    with a pause in it, not a floor grab, and the lexicon already accepts the repetition.
    """
    speaker = ordered[index].speaker_idx
    end = ordered[index].end
    nxt = next((o for o in ordered[index + 1:] if o.speaker_idx == speaker), None)
    if nxt is None or nxt.start - end > clearance:
        return False
    return nxt.kind is None


def find_backchannels(
    utterances,
    ipu_pause=0.3,
    max_duration=1.5,
    floor_gap=2.0,
    include_laughter=True,
    lexicon="extended",
    num_speakers=2,
    turn_clearance=0.0,
):
    """Classify every IPU in the dialogue, returning them in chronological order.

    An IPU is a backchannel when it is short (<= max_duration), passes the content gate, and did
    not take the floor:

      content: every word is a backchannel word or phrase (is_backchannel_text), or -- when
               include_laughter is set -- the unit is nothing but non-speech events, one of which
               is laughter.
      floor:   it overlaps an opposing speaker's IPU, or that speaker was talking immediately
               before and resumes immediately after (see _held_floor).

    The floor condition is what separates a backchannel from the same words used as a turn:
    "yeah" answering a question takes the floor, "yeah" over or between the partner's sentences
    does not.
    """
    per_speaker = {s: _build_ipus(utterances, s, ipu_pause) for s in range(num_speakers)}
    spans = {
        s: _IntervalSet([(w[0][0], w[-1][1]) for w in ipus]) for s, ipus in per_speaker.items()
    }

    def content_kind(words):
        tokens = [w[2] for w in words]
        lexical = [w for w in tokens if not is_non_speech(w)]
        if lexical:
            return "lexical" if is_backchannel_text(" ".join(lexical), lexicon) else None
        if include_laughter and any(w.strip().lower() == _LAUGHTER_TOKEN for w in tokens):
            return "laughter"
        return None

    ordered = []
    for speaker_idx, ipus in per_speaker.items():
        for words in ipus:
            start, end = words[0][0], words[-1][1]
            kind = content_kind(words) if end - start <= max_duration else None
            # The onset is the first *lexical* word, not simply the first: a "[noise] yeah" unit
            # marked on the bracket token would lose its <bc> marker outright if non-speech words
            # get dropped from the text channel. Falls back to the first word for a unit that is
            # all non-speech (a laughter backchannel).
            onset = next((w for w in words if not is_non_speech(w[2])), words[0])
            ordered.append(
                Ipu(
                    speaker_idx=speaker_idx,
                    start=start,
                    end=end,
                    text=" ".join(w[2] for w in words),
                    is_backchannel=False,  # filled in below, once the list is ordered
                    utt_idx=onset[3],
                    word_idx=onset[4],
                    kind=kind,
                )
            )
    ordered.sort(key=lambda ipu: ipu.start)

    # The floor test needs the chronological ordering, so acceptance is a second pass.
    result = []
    for index, ipu in enumerate(ordered):
        held = ipu.kind is not None and (
            any(s != ipu.speaker_idx and spans[s].overlaps(ipu.start, ipu.end) for s in spans)
            or _held_floor(ordered, index, floor_gap)
        )
        precedes = bool(held and turn_clearance > 0 and _precedes_own_turn(
            ordered, index, turn_clearance
        ))
        result.append(ipu._replace(
            is_backchannel=held, kind=ipu.kind if held else None, precedes_own_turn=precedes,
        ))
    return result


def _overlap_amount(host, start, end):
    return min(host["end_time"], end) - max(host["start_time"], start)


def relabel_backchannels(utterances, **find_backchannels_kwargs):
    """Re-derive `utterances[*].backchannels` from word alignments, via find_backchannels
    (lexicon + floor-preserving-overlap heuristic), instead of the dataset's own scripted labels.

    `utterances`: list of {speaker_idx, start_time, end_time, tts_text, words, backchannels}.
    `backchannels` may be a nested list of same-shaped dicts on each utterance it overlaps, or
    always-empty; either way it gets replaced.  Any extra keys on an utterance are preserved on
    whichever output entry it ends up as (main or nested).

    `find_backchannels_kwargs`: forwarded to find_backchannels (e.g. ipu_pause, max_duration,
    floor_gap, lexicon, include_laughter, turn_clearance) - defaults match its own.

    Returns a new list in the same shape as the input, with `backchannels` rebuilt from
    find_backchannels' own IPU classification rather than copied from the input.

    Known limitation: when find_backchannels merges several flat units into one wider Ipu (same
    speaker, gap < ipu_pause), no single unit's own recorded span contains that merged envelope,
    so those units conservatively fall back to "not backchannel" rather than risk mislabeling -
    a small, expected undercount, not a correctness bug in the common single-unit-per-Ipu case.
    """
    # Flatten: speaker_words()/find_backchannels ignore list order entirely (they re-sort every
    # speaker's words by timestamp internally), so main utterances and already-nested
    # backchannel events can simply be concatenated into one flat list of candidate speech units
    # - no reshaping needed, since a nested backchannel dict already has the same
    # {speaker_idx, words} shape find_backchannels expects from a top-level utterance.
    flat = list(utterances)
    for utt in utterances:
        flat.extend(utt.get("backchannels", []))

    ipus = find_backchannels(flat, **find_backchannels_kwargs)

    # Match each flat unit to the Ipu whose span it contains. An Ipu's [start, end] is derived
    # purely from the words it groups (see speaker_words/_build_ipus), which is <= a unit's own
    # recorded [start_time, end_time] whenever that unit's declared span includes any lead-in/
    # trail-off silence beyond its first/last word - so the unit contains the Ipu, not the other
    # way round, in the (overwhelmingly common) case where an Ipu doesn't span multiple units.
    def label_for(unit):
        speaker = unit["speaker_idx"]
        start, end = unit["start_time"], unit["end_time"]
        for ipu in ipus:
            if ipu.speaker_idx == speaker and start <= ipu.start + 1e-6 and end >= ipu.end - 1e-6:
                return ipu.is_backchannel
        return False  # no matching Ipu (e.g. empty words) - treat as not-a-backchannel

    is_bc = [label_for(unit) for unit in flat]

    # Units find_backchannels rejects become (or stay) standalone main utterances; scripted
    # backchannels dropped this way lose their nesting and reappear as their own turn, exactly
    # like a main utterance the dataset never flagged as backchannel in the first place.
    main_utterances = []
    for unit, bc_flag in zip(flat, is_bc):
        if bc_flag:
            continue
        new_unit = dict(unit)
        new_unit["backchannels"] = []
        # A demoted former-nested-backchannel has no uttr_type at all (that key only ever
        # existed on top-level utterances) - default it like any other ordinary turn.
        new_unit.setdefault("uttr_type", None)
        main_utterances.append(new_unit)

    # Units find_backchannels accepts (whether previously nested or previously a standalone
    # main utterance) get nested under whichever opposing-speaker main utterance overlaps them
    # most - the same "who is this backchannel over" question the original nesting encodes.
    for unit, bc_flag in zip(flat, is_bc):
        if not bc_flag:
            continue
        speaker, start, end = unit["speaker_idx"], unit["start_time"], unit["end_time"]
        candidates = [h for h in main_utterances if h["speaker_idx"] != speaker]
        if not candidates:
            continue
        host = max(candidates, key=lambda h: _overlap_amount(h, start, end))
        new_bc = dict(unit)
        new_bc.pop("backchannels", None)
        host["backchannels"].append(new_bc)

    main_utterances.sort(key=lambda u: u["start_time"])

    # create_text_stream derives the conversation's total duration from
    # utterances[-1]["end_time"] (see get_text_stream.py) - the last item in
    # list order, not a max over all items. If the utterance that used to be
    # temporally last got reclassified as a backchannel and nested inside an
    # earlier one, the new last main utterance ends earlier than the audio
    # actually does, silently shrinking that duration and inflating every
    # frame index downstream. Preserve the original boundary explicitly
    # rather than relying on sort order to happen to preserve it.
    if main_utterances:
        original_end_time = utterances[-1]["end_time"]
        main_utterances[-1]["end_time"] = max(
            main_utterances[-1]["end_time"], original_end_time
        )
    return main_utterances
