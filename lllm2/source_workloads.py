"""Fixed representative code snapshot used only as text, never executed."""
SOURCE = '''from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Transfer:
    account: str
    amount: Decimal
    reference: str


def parse_transfer(record):
    # Preserve the external reference for reconciliation.
    account = record["account"].strip()
    amount = Decimal(record["amount"])
    reference = record.get("reference", "manual")
    if not account:
        raise ValueError("account is required")
    if amount < 0:
        raise ValueError("amount must be positive")
    return Transfer(account, amount, reference)


def totals_by_account(transfers):
    totals = {}
    for transfer in transfers:
        previous = totals.get(transfer.account, Decimal("0"))
        totals[transfer.account] = previous + transfer.amount
    return totals


def export_rows(transfers):
    # Stable order keeps repeated exports easy to compare.
    ordered = sorted(transfers, key=lambda item: (item.account, item.reference))
    return [
        {"account": item.account, "amount": str(item.amount),
         "reference": item.reference}
        for item in ordered
    ]
'''
TASKS = {
    'source-copy': 'Copy the complete source excerpt below exactly. Preserve every comment, blank line, indentation, name and statement order. Return only the complete Python source, without explanations. Do not implement or improve anything.',
    'source-small-edit': 'In parse_transfer, change only `if amount < 0:` to `if amount <= 0:` so zero amounts are rejected too. Return the complete source excerpt with that single change. Preserve all other characters, including comments, blank lines, indentation, names and statement order. Return only the complete Python source, without explanations.',
}
EXPECTED = {key: SOURCE if key == 'source-copy' else SOURCE.replace('if amount < 0:', 'if amount <= 0:') for key in TASKS}

# Fixed unrelated reference modules mix repeated syntax and novel structures.
REFERENCE = '''# routing.py
from urllib.parse import urlsplit

def route_key(method, url):
    path = urlsplit(url).path.rstrip("/") or "/"
    return method.upper(), path

# retry.py
from time import monotonic

def remaining(deadline):
    return max(0.0, deadline - monotonic())

def retry_delay(attempt, ceiling=30):
    return min(ceiling, 2 ** attempt)

# inventory.py
from collections import Counter

def missing_items(requested, available):
    return dict(Counter(requested) - Counter(available))

def batches(items, size):
    if size < 1:
        raise ValueError("size must be positive")
    for start in range(0, len(items), size):
        yield items[start:start + size]

'''


def output_budget(workload, requested):
    return max(2048, requested) if workload in TASKS else requested


def source_prompt(bench, settings, workload, tokens, timeout):
    import hashlib
    marker = 'LLLM2_REFERENCE_CONTEXT_PLACEHOLDER'
    task = TASKS[workload] + '\n\nSOURCE EXCERPT:\n' + SOURCE + '\nEND SOURCE EXCERPT\nReturn the complete source now.'
    text = 'Unrelated reference context, not part of the requested output:\n' + marker + '\n\n' + task
    body = dict(messages=[dict(role='user', content=text)], add_generation_prompt=True)
    if settings.effort != 'default':
        body['reasoning_effort'] = settings.effort
    formatted = bench.req('/apply-template', body, timeout)['prompt']
    if formatted.count(marker) != 1:
        raise RuntimeError('Template did not preserve source context placeholder.')
    prefix, suffix = formatted.split(marker)
    def tokenize(content, special=False):
        return bench.req('/tokenize', dict(content=content, add_special=special, parse_special=True), timeout)['tokens']
    head, tail = tokenize(prefix, True), tokenize(suffix)
    room = tokens - len(head) - len(tail)
    if room < 0:
        raise ValueError(f'Source workload requires at least {len(head)+len(tail)} input tokens to preserve the complete excerpt and instructions.')
    reference = tokenize(REFERENCE)
    if not reference:
        raise RuntimeError('Could not tokenize reference context.')
    padding = (reference * ((room + len(reference) - 1) // len(reference)))[:room]
    actual = head + padding + tail
    return actual, dict(generator='source-snapshot-v1', workload=workload, source=SOURCE,
                        source_sha256=hashlib.sha256(SOURCE.encode()).hexdigest(), instruction=TASKS[workload],
                        reference_sha256=hashlib.sha256(REFERENCE.encode()).hexdigest(),
                        padding_tokens=room, token_ids=actual, expected_output=EXPECTED[workload],
                        limitation='Fixed representative source and repeated unrelated reference modules; narrow adherence check, not general coding quality.')


def adherence(response, expected):
    import hashlib
    import re
    raw = response.get('content', '')
    payload = raw
    thinking = None
    if '</think>' in payload:
        thinking, payload = payload.rsplit('</think>', 1)
    payload = payload.strip()
    fenced = re.fullmatch(r'```(?:python|py)?\s*\n(.*?)\n```', payload, re.S)
    if fenced:
        payload = fenced[1].strip()
    limited = bool(response.get('truncated') or response.get('stopped_limit') or response.get('stop_type') == 'limit')
    matches = payload == expected.strip()
    return dict(status='passed' if matches and not limited else 'failed', exact_payload_match=matches,
                output_limit_reached=limited, thinking_characters=len(thinking) if thinking is not None else None,
                thinking_tokens=None, final_payload=payload,
                expected_sha256=hashlib.sha256(expected.strip().encode()).hexdigest(),
                actual_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                policy='Compare complete source after final </think> and optional single Python code fence; ignore only outer whitespace. Raw output retained. No code execution or semantic scoring.')
