# -*- coding: utf-8 -*-
import json
import hashlib
from pathlib import Path


def load_prompt(path):
    return Path(path).read_text(encoding="utf-8")


def render_prompt(template, industry, title, text, max_text_chars):
    title = "" if title is None else str(title)
    text = "" if text is None else str(text)
    text = text[:int(max_text_chars)]
    return template.replace("{{industry}}", str(industry)).replace("{{title}}", title).replace("{{text}}", text)


def make_task_id(article_id, industry):
    raw = "%s|%s" % (str(article_id), str(industry))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def append_jsonl(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_done_ids(path):
    p = Path(path)
    done = set()
    if not p.exists():
        return done
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("ok") and obj.get("task_id"):
                    done.add(obj["task_id"])
            except Exception:
                pass
    return done


def parse_json_object(text):
    text = str(text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    l = text.find("{")
    r = text.rfind("}")
    if l >= 0 and r > l:
        obj = json.loads(text[l:r + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("model output is not a JSON object")


def validate_label(obj):
    direction = int(obj["direction"])
    if direction not in (-1, 0, 1):
        raise ValueError("invalid direction")
    confidence = float(obj["confidence"])
    relevance = float(obj["relevance"])
    strength = float(obj["policy_strength"])
    for name, value in [("confidence", confidence), ("relevance", relevance), ("policy_strength", strength)]:
        if value < 0.0 or value > 1.0:
            raise ValueError("%s outside [0,1]" % name)
    policy_type = str(obj.get("policy_type", "other")).strip()
    allowed = {"support", "stimulus", "subsidy", "tax", "investment", "demand", "supply", "reform", "regulation", "restriction", "risk", "other"}
    if policy_type not in allowed:
        policy_type = "other"
    return {
        "direction": direction,
        "confidence": confidence,
        "relevance": relevance,
        "policy_strength": strength,
        "policy_type": policy_type,
        "reason": str(obj.get("reason", "")).strip()[:200],
    }
