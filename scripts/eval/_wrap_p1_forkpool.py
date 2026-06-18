"""One-shot source transform: wrap the serial P1 enumerate block in
beam_search_v7.py under `if n_workers > 1: <forkpool> else: <serial verbatim>`.

The serial body is indented by 4 spaces (whitespace-only change, code identical)
so the n_workers==1 path runs the original optimized code unchanged. Finds the
block by content markers (robust to line drift). Idempotent guard: aborts if the
wrapper marker is already present.
"""
import sys

PATH = 'scripts/eval/beam_search_v7.py'
START = '        tasks = []  # (parent_idx, target, valid)\n'
END = '                    tasks.append((parent_idx, target, valid))\n'
GUARD = '        if n_workers > 1:\n'

with open(PATH) as f:
    lines = f.readlines()

# idempotency: if the guard line already sits where START is, abort.
try:
    si = lines.index(START)
except ValueError:
    sys.exit('START marker not found — aborting (no change).')
if si >= 1 and lines[si - 1] == GUARD.replace('\n', '') + '\n':
    sys.exit('Wrapper already present — aborting (no change).')

# find END at-or-after START
ei = None
for j in range(si, len(lines)):
    if lines[j] == END:
        ei = j
        break
if ei is None:
    sys.exit('END marker not found after START — aborting (no change).')

block = lines[si:ei + 1]
# every block line is non-empty here; indent each by 4 spaces (blank lines, if
# any, stay blank).
indented = [('    ' + ln) if ln.strip() else ln for ln in block]

wrapper_head = [
    '        if n_workers > 1:\n',
    '            # Parallel enumerate: fork workers over disjoint parent slices.\n',
    '            tasks, _aux_updates = _forkpool_enumerate(beam, n_workers)\n',
    '            for _pi, _na in _aux_updates.items():\n',
    '                beam[_pi] = beam[_pi]._replace(aux_flat=_na)\n',
    '        else:\n',
    '            # === serial enumerate path — original code, indent-only change ===\n',
]

new_lines = lines[:si] + wrapper_head + indented + lines[ei + 1:]

with open(PATH, 'w') as f:
    f.writelines(new_lines)

print(f'Wrapped P1 block: lines {si + 1}..{ei + 1} '
      f'({ei - si + 1} lines) indented + {len(wrapper_head)} wrapper lines inserted.')
