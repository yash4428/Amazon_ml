"""Write Documentation_template.md for the genuine-rebuild package: METHODOLOGY.md + a header describing exactly what
the package output is, with the measured numbers (OOF, candidate size, agreement with the best uploaded file).

    python make_rebuild_doc.py METHODOLOGY.md COMPARE_TXT BLEND_TXT CANDIDATE_TSV OUT_MD
"""
import sys


def main():
    """Prepend the rebuild note (numbers filled from the overnight run outputs) to the methodology."""
    meth, compare, blend, cand, out = sys.argv[1:6]
    comp = open(compare).read().strip().splitlines()
    comp = comp[-1] if comp else "(comparison unavailable)"
    bl = open(blend).read().strip().splitlines()
    bl = bl[-1] if bl else "(single run)"
    n_pairs, n_s1 = 0, 0
    with open(cand) as f:
        next(f)
        for line in f:
            n_s1 += 1
            ids = line.rstrip("\n").split("\t", 1)[1] if "\t" in line else ""
            n_pairs += len([x for x in ids.split(",") if x])
    note = f"""> **This package = genuine rebuild of the teammate's approach**, regenerated end-to-end from the raw data by the
> code in `code/business_entity_resolution/` (README §C): our pipeline + teammate-style name-only and address-only
> candidate generators (`--extra-gens`, reverse-engineered from her output file), 3 seeds, blend, label-free rules.
> It uses no external input and no leaderboard feedback beyond what the methodology describes.
> **It was produced after the leaderboard closed and was never uploaded**, so it has no public score. The team's best
> uploaded file (0.989378, §5.3) was built on a teammate model whose source code was lost.
> - OOF (3-seed blend on common train S1): `{bl}`
> - Candidate set (exact stage-1 output scored by the model): {n_pairs:,} pairs, {n_pairs / max(n_s1, 1):.1f} per S1
> - Agreement with the best uploaded file's teammate base (her matches used as reference; "cover" = share of her
>   matches inside our candidate set): `{comp}`

"""
    s = open(meth).read()
    s = s.replace("---\n\n## 1. Executive Summary", note + "---\n\n## 1. Executive Summary", 1)
    open(out, "w").write(s)
    print("wrote", out)


if __name__ == "__main__":
    main()
