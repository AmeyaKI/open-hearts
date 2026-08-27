"""CI probe: compare a freshly-run R0 block 0 against the committed local
reference for exact (string-formatted) bitwise equality.

Reference: ci/r0_block0_reference.txt, generated locally on an arm64 Mac
2026-08-26 by `run_exploit_eval.py --rows R0 --deals 25 --workers 8 --fast`
(the banked results/exploit_R0_partial.txt only had blocks 1-5, not block 0,
so this fixture was generated fresh -- both sides come from the same code
revision, not a pre-existing historical bank).

CI side: results/exploit_R0_partial.txt produced by the matching run in this
workflow (fresh checkout, no prior partial file, so block 0 lands there).

Never raises/exits non-zero on a mismatch -- the verdict is printed and the
job must continue on to the R1v timing probe regardless.
"""
import json
import sys


def load_block(path):
    with open(path) as f:
        lines = [ln for ln in f if ln.strip()]
    blocks = {}
    for ln in lines:
        tag, payload = ln.split(" ", 1)
        row, block_idx = tag.split("@")
        assert payload.startswith("J")
        blocks[(row, int(block_idx))] = json.loads(payload[1:])
    return blocks


def main():
    ref_path = "ci/r0_block0_reference.txt"
    ci_path = "results/exploit_R0_partial.txt"

    ref_blocks = load_block(ref_path)
    ci_blocks = load_block(ci_path)

    ref = ref_blocks.get(("R0", 0))
    ci = ci_blocks.get(("R0", 0))

    if ref is None:
        print("BITWISE: DIFFERS deal=ALL local=<missing R0@0 in reference> ci=n/a")
        return
    if ci is None:
        print("BITWISE: DIFFERS deal=ALL local=n/a ci=<missing R0@0 in CI output>")
        return

    mismatches = 0
    total = 0
    for key in ("values", "champ"):
        ref_arr = ref[key]
        ci_arr = ci.get(key, [])
        n = max(len(ref_arr), len(ci_arr))
        for i in range(n):
            total += 1
            rv = repr(ref_arr[i]) if i < len(ref_arr) else "<missing>"
            cv = repr(ci_arr[i]) if i < len(ci_arr) else "<missing>"
            deal = 100000 + i
            if rv != cv:
                mismatches += 1
                print(f"BITWISE: DIFFERS deal={deal} field={key} "
                      f"local={rv} ci={cv}")

    if mismatches == 0:
        print(f"BITWISE: IDENTICAL ({total} values/champ entries across "
              f"25 deals, exact repr() match)")
    else:
        print(f"BITWISE: {mismatches}/{total} entries differ (see lines above)")

    # Secondary canary: diag floats (summation-order / libm sensitive; the
    # coarse quarter-point `values`/`champ` above could coincidentally match
    # even if the underlying float path diverges).
    diag_ref = ref.get("diag", {})
    diag_ci = ci.get("diag", {})
    diag_mismatches = 0
    diag_total = 0
    for k in sorted(set(diag_ref) | set(diag_ci)):
        rv = diag_ref.get(k)
        cv = diag_ci.get(k)
        if isinstance(rv, list):
            for i, (a, b) in enumerate(zip(rv, cv or [])):
                diag_total += 1
                if repr(a) != repr(b):
                    diag_mismatches += 1
                    print(f"DIAG: DIFFERS key={k}[{i}] local={a!r} ci={b!r}")
        else:
            diag_total += 1
            if repr(rv) != repr(cv):
                diag_mismatches += 1
                print(f"DIAG: DIFFERS key={k} local={rv!r} ci={cv!r}")

    if diag_mismatches == 0:
        print(f"DIAG: IDENTICAL ({diag_total} diag entries, exact repr() match)")
    else:
        print(f"DIAG: {diag_mismatches}/{diag_total} diag entries differ")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"BITWISE: DIFFERS deal=ALL local=<comparison script error> ci={e!r}")
        sys.exit(0)
